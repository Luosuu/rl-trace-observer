"""Discover and read per-process trace artifacts.

Every reader returns a :class:`TraceSource` whose event timestamps are relative
to an integer Unix-epoch anchor, so sources from different profilers can be
placed on one timeline without losing precision:

* RL-Insight JSONL written by this package: ``ts`` is epoch µs.
* Torch Profiler (Kineto) Chrome traces: ``ts`` is relative to the top-level
  ``baseTimeNanoseconds``.
* VizTracer reports: ``ts`` is relative to ``viztracer_metadata.baseTimeNanoseconds``.
"""

import gzip
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RL_INSIGHT = "rl_insight"
TORCH = "torch"
VIZTRACER = "viztracer"
PROCESS_RECORD = "process_record"

# rl-trace-process-<host>-pid-<pid>.json, written by process_record.register_artifact.
_PROCESS_RECORD_FILE = re.compile(r"^rl-trace-process-(?P<host>.+)-pid-(?P<pid>\d+)\.json$")
# rl-insight-<host>-pid-<pid>.chrome.jsonl, written by ChromeTraceJsonlClient.
_RL_INSIGHT_FILE = re.compile(r"^rl-insight-(?P<host>.+)-pid-(?P<pid>\d+)\.chrome\.jsonl$")
# verl.utils.profiler.torch_profile.build_trace_basename plus optional _partN/_mbN suffix.
_TORCH_FILE = re.compile(
    r"^(?P<label>.+)_pid(?P<pid>\d+)_(?P<timestamp>\d{17})(?P<part>_(?:part|mb)[\w-]+)?\.json(?:\.gz)?$"
)
# step-<step>-role-<role>-rank-<rank>-pid-<pid>.viztracer.json, written by VizTracerObserver.
_VIZTRACER_FILE = re.compile(
    r"^step-(?P<step>.+)-role-(?P<role>.+)-rank-(?P<rank>\d+)-pid-(?P<pid>\d+)\.viztracer\.json$"
)

# Kineto's own bookkeeping pseudo-processes: the profiling window ("Spans"),
# iteration markers ("Traces") and window-end instants (empty pid). GPU devices
# use integer pids and are kept.
_KINETO_BOOKKEEPING_PIDS = frozenset({"Spans", "Traces", ""})
_TORCH_RANK = re.compile(r"_rank(?P<rank>\d+)(?:-of-\d+)?(?:_|$)")


class ArtifactError(ValueError):
    """An artifact that cannot be read, e.g. truncated or missing its time anchor."""


@dataclass
class TraceSource:
    """One per-process trace artifact."""

    kind: str
    path: Path
    title: str
    base_ns: int
    events: list[dict[str, Any]]
    os_pid: int
    rank: int | None = None
    host: str | None = None
    labels: dict[str, str] = field(default_factory=dict)
    # False when the artifact was cut short, e.g. a worker killed mid-write.
    complete: bool = True
    # Key of the OS process this artifact belongs to ("<host>:<pid>"), once known.
    process: str | None = None


def classify(path: Path) -> str | None:
    """Return the source kind for an artifact filename, or ``None``."""
    name = path.name
    if _PROCESS_RECORD_FILE.match(name):
        return PROCESS_RECORD
    if _RL_INSIGHT_FILE.match(name):
        return RL_INSIGHT
    if _VIZTRACER_FILE.match(name):
        return VIZTRACER
    if _TORCH_FILE.match(name):
        return TORCH
    return None


def discover(paths: Iterable[Path]) -> list[tuple[str, Path]]:
    """Find artifacts and process records in the given files and directories (recursively)."""
    found: dict[Path, str] = {}
    for path in paths:
        candidates = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for candidate in candidates:
            kind = classify(candidate)
            if kind is None:
                if path.is_file():
                    raise ValueError(f"Unrecognized trace artifact: {candidate}")
                logger.debug("Skipping unrecognized file %s", candidate)
                continue
            found[candidate.resolve()] = kind
    return sorted(((kind, path) for path, kind in found.items()), key=lambda item: str(item[1]))


def read_source(kind: str, path: Path) -> TraceSource:
    if kind == RL_INSIGHT:
        return read_rl_insight_jsonl(path)
    if kind == TORCH:
        return read_torch_trace(path)
    if kind == VIZTRACER:
        return read_viztracer_trace(path)
    raise ValueError(f"Not a trace source kind: {kind}")


def read_rl_insight_jsonl(path: Path) -> TraceSource:
    match = _RL_INSIGHT_FILE.match(path.name)
    if match is None:
        raise ValueError(f"Not an RL-Insight JSONL artifact: {path}")

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    events = []
    complete = True
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as error:
            # The writer appends one line per event, so only the final line can
            # be cut short by a worker that was killed mid-write.
            if index == len(lines) - 1:
                complete = False
                continue
            raise ArtifactError(f"Corrupt line {index + 1} in {path}") from error

    # ts is absolute epoch µs. Re-anchor on a whole microsecond like the other
    # sources, since epoch nanoseconds exceed float64's exact integer range.
    base_us = int(min((float(event["ts"]) for event in events if "ts" in event), default=0))
    for event in events:
        if "ts" in event:
            event["ts"] = float(event["ts"]) - base_us

    host, pid = match["host"], int(match["pid"])
    return TraceSource(
        kind=RL_INSIGHT,
        path=path,
        title=f"RL-Insight {host} pid {pid}",
        base_ns=base_us * 1000,
        events=events,
        os_pid=pid,
        host=host,
        complete=complete,
        process=f"{host}:{pid}",
    )


def _load_json(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"Cannot read {path}: {error}") from error


def _anchor(value: object, path: Path, field_name: str) -> int:
    if value is None:
        raise ArtifactError(f"{path} has no {field_name}; cannot place it on an absolute timeline")
    return int(value)


def read_torch_trace(path: Path) -> TraceSource:
    match = _TORCH_FILE.match(path.name)
    if match is None:
        raise ValueError(f"Not a VERL Torch Profiler trace: {path}")

    data = _load_json(path)
    label, pid = match["label"], int(match["pid"])
    events = [event for event in data.get("traceEvents", []) if event.get("pid") not in _KINETO_BOOKKEEPING_PIDS]
    # Kineto records the torch.distributed rank once the process group exists;
    # VERL's filename carries the rank it was given otherwise.
    rank = data.get("distributedInfo", {}).get("rank")
    if rank is None and (rank_match := _TORCH_RANK.search(label)):
        rank = int(rank_match["rank"])
    return TraceSource(
        kind=TORCH,
        path=path,
        title=f"Torch {label} pid {pid}",
        base_ns=_anchor(data.get("baseTimeNanoseconds"), path, "baseTimeNanoseconds"),
        events=events,
        os_pid=pid,
        rank=rank,
        labels={"label": label},
    )


def read_viztracer_trace(path: Path) -> TraceSource:
    match = _VIZTRACER_FILE.match(path.name)
    if match is None:
        raise ValueError(f"Not an RL Trace Observer VizTracer report: {path}")

    data = _load_json(path)
    metadata = data.get("viztracer_metadata", {})
    step, role, rank, pid = match["step"], match["role"], int(match["rank"]), int(match["pid"])
    return TraceSource(
        kind=VIZTRACER,
        path=path,
        title=f"VizTracer {role} step {step} rank {rank} pid {pid}",
        base_ns=_anchor(metadata.get("baseTimeNanoseconds"), path, "viztracer_metadata.baseTimeNanoseconds"),
        events=list(data.get("traceEvents", [])),
        os_pid=pid,
        rank=rank,
        labels={"step": step, "role": role},
    )
