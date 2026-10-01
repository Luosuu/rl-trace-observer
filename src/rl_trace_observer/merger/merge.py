"""Merge per-process trace sources into one Chrome Trace for Perfetto.

The merger never reuses a source's pid/tid: OS pids collide across nodes and
Kineto also emits device indices and strings such as ``"Spans"`` as pids.

* Events a source recorded for its own OS process go to one synthetic pid per
  OS process (``source.process``), so the RL-Insight lanes, Torch threads and
  VizTracer threads of one worker appear together.
* Any other pid in a source (e.g. a Kineto GPU device) gets its own synthetic
  pid, named after the process that recorded it.
* A TokenSpeed scheduler rank gets one synthetic pid for its VizTracer report
  and Proton trace, named after the server actor that registered them.
* Threads get trace-wide unique synthetic tids, named with their source kind.
* Flow and async event ids are renumbered per source so they cannot connect
  events of unrelated processes. The one exception links a TokenSpeed rank's
  two files: its VizTracer report starts a flow at every Python scope whose id
  is the Proton CPU scope's ``scope_id``, and the merger ends it on that scope,
  as ``tokenspeed merge-traces`` does.
"""

import itertools
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .manifest import Process
from .sources import RL_INSIGHT, TOKENSPEED_PROTON, TOKENSPEED_VIZTRACER, TORCH, VIZTRACER, TraceSource

_KIND_ORDER = {RL_INSIGHT: 0, TORCH: 1, VIZTRACER: 2, TOKENSPEED_VIZTRACER: 3, TOKENSPEED_PROTON: 4}
_KIND_LABEL = {
    RL_INSIGHT: "RL-Insight",
    TORCH: "Torch",
    VIZTRACER: "VizTracer",
    TOKENSPEED_VIZTRACER: "VizTracer",
    TOKENSPEED_PROTON: "Proton",
}
_TOKENSPEED = frozenset({TOKENSPEED_VIZTRACER, TOKENSPEED_PROTON})
# The flow TokenSpeed's VizTracer report starts at each Python scope that Proton
# also records, and the Proton argument naming the scope (tokenspeed.cli.trace_merge).
_SCOPE_FLOW_NAME = "viztracer->proton"
_SCOPE_FLOW_CATEGORY = "tokenspeed.proton"
_SCOPE_ID_ARG = "scope_id"
# Chrome flow (s/t/f) and async (b/n/e, legacy S/T/F) phases carry an id that
# links events; ids are only unique within the file that produced them.
_ID_PHASES = frozenset("stfbneSTF")


class EmptyTraceError(ValueError):
    """No source has a timed event, e.g. every process ran without writing spans."""


@dataclass
class MergeResult:
    trace: dict[str, Any]
    global_base_ns: int
    warnings: list[str] = field(default_factory=list)


def _metadata(name: str, pid: int, tid: int, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "ph": "M", "pid": pid, "tid": tid, "args": args}


def _event_start_ns(source: TraceSource, event: dict[str, Any]) -> int | None:
    if event.get("ph") == "M" or "ts" not in event:
        return None
    return source.base_ns + round(float(event["ts"]) * 1000)


class _Namespace:
    """Synthetic pids and tids shared by all sources of one merge."""

    def __init__(self, processes: Mapping[str, Process]):
        self._processes = processes
        self._pids: dict[tuple, int] = {}
        self._next_pid = itertools.count(1)
        self._next_tid = itertools.count(1)
        # Flow ids shared by the two files of a TokenSpeed rank, by (pair, id).
        self.shared_flow_ids: dict[tuple, int] = {}
        self.metadata: list[dict[str, Any]] = []

    def pid(self, key: tuple, name: str, labels: dict[str, Any]) -> int:
        if key not in self._pids:
            pid = self._pids[key] = next(self._next_pid)
            self.metadata.append(_metadata("process_name", pid, 0, {"name": name}))
            self.metadata.append(_metadata("process_sort_index", pid, 0, {"sort_index": pid}))
            text = ", ".join(f"{k}={v}" for k, v in labels.items() if v is not None)
            self.metadata.append(_metadata("process_labels", pid, 0, {"labels": text}))
        return self._pids[key]

    def new_tid(self, pid: int, name: str) -> int:
        # Perfetto's JSON importer identifies threads by tid alone, like OS
        # tids, so tids must be unique across processes, not per process.
        tid = next(self._next_tid)
        self.metadata.append(_metadata("thread_name", pid, tid, {"name": name}))
        self.metadata.append(_metadata("thread_sort_index", pid, tid, {"sort_index": tid}))
        return tid

    def process(self, key: str | None) -> Process | None:
        return self._processes.get(key) if key is not None else None


class _SourceRemapper:
    """Maps one source's pids, tids and flow ids into the merged namespace."""

    def __init__(self, source: TraceSource, namespace: _Namespace, ids: itertools.count):
        self._source = source
        self._namespace = namespace
        self._next_id = ids
        self._pids: dict[str, int] = {}
        self._tids: dict[tuple[str, str], int] = {}
        self._ids: dict[str, int] = {}
        self._process_names: dict[str, str] = {}
        self._thread_names: dict[tuple[str, str], str] = {}
        for event in source.events:
            if event.get("ph") != "M" or not (name := event.get("args", {}).get("name")):
                continue
            if event.get("name") == "process_name":
                self._process_names[str(event.get("pid"))] = name
            elif event.get("name") == "thread_name":
                self._thread_names[(str(event.get("pid")), str(event.get("tid")))] = name

    def _is_own_process(self, original: str) -> bool:
        # RL-Insight writes the process id as a string attribute; profilers use the OS pid.
        return self._source.kind == RL_INSIGHT or original == str(self._source.os_pid)

    def pid(self, original: object) -> int:
        key = str(original)
        if key not in self._pids:
            source = self._source
            process = self._namespace.process(source.process)
            owner = process.title if process else source.title
            if source.kind in _TOKENSPEED:
                # Everything a scheduler rank recorded: its own pid in the
                # VizTracer report and Proton's synthetic pid 0.
                if process:
                    namespace_key = ("tokenspeed", process.key, source.rank_tag)
                    name = f"{process.title} · {source.rank_tag}"
                    labels = {"host": process.hostname, "actor": process.actor_name, "rank_tag": source.rank_tag}
                else:
                    namespace_key = ("tokenspeed", source.pair)
                    name = f"TokenSpeed {source.labels.get('profile_id')} {source.rank_tag}"
                    labels = {"rank_tag": source.rank_tag}
            elif self._is_own_process(key):
                if process:
                    namespace_key = ("process", process.key)
                    labels = {"host": process.hostname, "pid": process.os_pid, "rank": process.rank}
                    labels["actor"] = process.actor_name
                else:
                    namespace_key = ("source", str(source.path), key)
                    labels = {"file": source.path.name}
                name = owner
            else:
                namespace_key = ("source", str(source.path), key)
                name = f"{owner} · {self._process_names.get(key) or key}"
                labels = {"file": source.path.name}
            self._pids[key] = self._namespace.pid(namespace_key, name, labels)
        return self._pids[key]

    def tid(self, original_pid: object, original: object) -> int:
        key = (str(original_pid), str(original))
        if key not in self._tids:
            thread = self._thread_names.get(key) or str(original) or "unassigned"
            name = f"{_KIND_LABEL.get(self._source.kind, self._source.kind)} · {thread}"
            self._tids[key] = self._namespace.new_tid(self.pid(original_pid), name)
        return self._tids[key]

    def flow_id(self, original: object) -> int:
        if isinstance(original, tuple):
            # A scope flow from _link_scope_flows, shared with the partner file.
            ids, key = self._namespace.shared_flow_ids, (self._source.pair, original)
        else:
            ids, key = self._ids, str(original)
        if key not in ids:
            ids[key] = next(self._next_id)
        return ids[key]


def _is_scope_flow_start(event: dict[str, Any]) -> bool:
    return event.get("ph") == "s" and event.get("name") == _SCOPE_FLOW_NAME and event.get("cat") == _SCOPE_FLOW_CATEGORY


def _scope_id(event: dict[str, Any]) -> int | None:
    args = event.get("args")
    value = args.get(_SCOPE_ID_ARG) if isinstance(args, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _link_scope_flows(sources: list[TraceSource]) -> dict[int, list[dict[str, Any]]]:
    """Copy each TokenSpeed rank's VizTracer report with its scope flows bound to its Proton trace.

    Returns, by ``id()`` of the Proton source, the flow ends to add to it. Flow
    starts whose scope Proton did not record (e.g. a profile without Proton)
    are dropped: they would point at nothing. Both ends get the same id,
    ``("scope", <scope_id>)``, which no profiler writes, so they are renumbered
    together and apart from every other flow.
    """
    proton = {source.pair: source for source in sources if source.kind == TOKENSPEED_PROTON}
    ends: dict[int, list[dict[str, Any]]] = {}
    for index, source in enumerate(sources):
        if source.kind != TOKENSPEED_VIZTRACER or not any(map(_is_scope_flow_start, source.events)):
            continue
        partner = proton.get(source.pair)
        scopes = {}
        for event in partner.events if partner else []:
            if (scope := _scope_id(event)) is not None and event.get("ph") != "M" and "ts" in event:
                scopes[scope] = event
        events = []
        for event in source.events:
            if not _is_scope_flow_start(event):
                events.append(event)
            elif (scope := event.get("id")) in scopes:
                events.append({**event, "id": ("scope", scope)})
                target = scopes[scope]
                end = {"name": _SCOPE_FLOW_NAME, "cat": _SCOPE_FLOW_CATEGORY, "ph": "f", "bp": "e"}
                end.update(ts=target["ts"], pid=target.get("pid"), tid=target.get("tid"), id=("scope", scope))
                ends.setdefault(id(partner), []).append(end)
        sources[index] = replace(source, events=events)
    return ends


def merge_sources(sources: Iterable[TraceSource], processes: Mapping[str, Process] | None = None) -> MergeResult:
    """Merge ``sources``; ``processes`` (from the session manifest) names linked OS processes."""
    ordered = sorted(sources, key=lambda source: (_KIND_ORDER.get(source.kind, 99), source.title, str(source.path)))
    scope_flow_ends = _link_scope_flows(ordered)
    starts = [_event_start_ns(source, event) for source in ordered for event in source.events]
    starts = [start for start in starts if start is not None]
    if not starts:
        raise EmptyTraceError("No timed events in any source")
    global_base_ns = min(starts)

    warnings: list[str] = []
    events: list[dict[str, Any]] = []
    namespace = _Namespace(processes or {})
    ids = itertools.count(1)

    for source in ordered:
        remapper = _SourceRemapper(source, namespace, ids)
        offset_us = (source.base_ns - global_base_ns) / 1000
        dropped = 0
        for event in itertools.chain(source.events, scope_flow_ends.get(id(source), [])):
            if event.get("ph") == "M":
                continue
            if "ts" not in event or float(event.get("dur", 0)) < 0:
                dropped += 1
                continue
            merged = dict(event)
            merged["pid"] = remapper.pid(event.get("pid"))
            merged["tid"] = remapper.tid(event.get("pid"), event.get("tid"))
            merged["ts"] = round(offset_us + float(event["ts"]), 3)
            if event.get("ph") in _ID_PHASES and "id" in event:
                merged["id"] = remapper.flow_id(event["id"])
            events.append(merged)
        if dropped:
            warnings.append(f"{source.path.name}: dropped {dropped} events without a timestamp or with negative dur")

    trace = {
        "traceEvents": namespace.metadata + events,
        "displayTimeUnit": "ms",
        "otherData": {
            "rl_trace_observer.global_base_time_ns": str(global_base_ns),
            "rl_trace_observer.sources": str(len(ordered)),
        },
    }
    return MergeResult(trace=trace, global_base_ns=global_base_ns, warnings=warnings)
