"""Merge per-process trace sources into one Chrome Trace for Perfetto.

The merger never reuses a source's pid/tid: OS pids collide across nodes and
Kineto also emits device indices and strings such as ``"Spans"`` as pids. Each
``(source, pid)`` gets a synthetic numeric pid with ``process_name`` metadata,
and each thread within it a trace-wide unique synthetic tid with
``thread_name`` metadata. Flow
and async event ids are renumbered per source so they cannot connect events of
unrelated processes.
"""

import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .sources import RL_INSIGHT, TORCH, VIZTRACER, TraceSource

_KIND_ORDER = {RL_INSIGHT: 0, TORCH: 1, VIZTRACER: 2}
# Chrome flow (s/t/f) and async (b/n/e, legacy S/T/F) phases carry an id that
# links events; ids are only unique within the file that produced them.
_ID_PHASES = frozenset("stfbneSTF")


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


class _SourceRemapper:
    """Maps one source's pids, tids and flow ids into the merged namespace."""

    def __init__(
        self, source: TraceSource, pids: itertools.count, tids: itertools.count, ids: itertools.count, metadata: list
    ):
        self._source = source
        self._next_pid = pids
        self._next_tid = tids
        self._next_id = ids
        self._metadata = metadata
        self._pids: dict[str, int] = {}
        self._tids: dict[tuple[int, str], int] = {}
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

    def pid(self, original: object) -> int:
        key = str(original)
        if key not in self._pids:
            pid = self._pids[key] = next(self._next_pid)
            source = self._source
            if source.kind == RL_INSIGHT or key == source.labels.get("pid"):
                # The profiled process itself: its recorded name is only the
                # process title at export time (e.g. a Ray method name).
                name = source.title
            else:
                name = f"{source.title} · {self._process_names.get(key) or key}"
            labels = ", ".join(f"{k}={v}" for k, v in {**source.labels, "file": source.path.name}.items())
            self._metadata.append(_metadata("process_name", pid, 0, {"name": name}))
            self._metadata.append(_metadata("process_sort_index", pid, 0, {"sort_index": pid}))
            self._metadata.append(_metadata("process_labels", pid, 0, {"labels": labels}))
        return self._pids[key]

    def tid(self, pid: int, original_pid: object, original: object) -> int:
        key = (pid, str(original))
        if key not in self._tids:
            # Perfetto's JSON importer identifies threads by tid alone, like OS
            # tids, so tids must be unique across processes, not per process.
            tid = self._tids[key] = next(self._next_tid)
            name = self._thread_names.get((str(original_pid), str(original))) or str(original) or "unassigned"
            self._metadata.append(_metadata("thread_name", pid, tid, {"name": name}))
            self._metadata.append(_metadata("thread_sort_index", pid, tid, {"sort_index": tid}))
        return self._tids[key]

    def flow_id(self, original: object) -> int:
        key = str(original)
        if key not in self._ids:
            self._ids[key] = next(self._next_id)
        return self._ids[key]


def merge_sources(sources: Iterable[TraceSource]) -> MergeResult:
    ordered = sorted(sources, key=lambda source: (_KIND_ORDER.get(source.kind, 99), source.title, str(source.path)))
    starts = [_event_start_ns(source, event) for source in ordered for event in source.events]
    starts = [start for start in starts if start is not None]
    if not starts:
        raise ValueError("No timed events in any source")
    global_base_ns = min(starts)

    warnings: list[str] = []
    metadata: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    pids, tids, ids = itertools.count(1), itertools.count(1), itertools.count(1)

    for source in ordered:
        remapper = _SourceRemapper(source, pids, tids, ids, metadata)
        offset_us = (source.base_ns - global_base_ns) / 1000
        dropped = 0
        for event in source.events:
            if event.get("ph") == "M":
                continue
            if "ts" not in event or float(event.get("dur", 0)) < 0:
                dropped += 1
                continue
            merged = dict(event)
            merged["pid"] = remapper.pid(event.get("pid"))
            merged["tid"] = remapper.tid(merged["pid"], event.get("pid"), event.get("tid"))
            merged["ts"] = round(offset_us + float(event["ts"]), 3)
            if event.get("ph") in _ID_PHASES and "id" in event:
                merged["id"] = remapper.flow_id(event["id"])
            events.append(merged)
        if dropped:
            warnings.append(f"{source.path.name}: dropped {dropped} events without a timestamp or with negative dur")

    trace = {
        "traceEvents": metadata + events,
        "displayTimeUnit": "ms",
        "otherData": {
            "rl_trace_observer.global_base_time_ns": str(global_base_ns),
            "rl_trace_observer.sources": str(len(ordered)),
        },
    }
    return MergeResult(trace=trace, global_base_ns=global_base_ns, warnings=warnings)
