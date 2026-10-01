"""Build the session manifest: which artifacts exist, whose they are, what is wrong.

Every process that writes an artifact registers it in its process record
(``rl-trace-process-<host>-pid-<pid>.json``): RL-Insight JSONL and VizTracer
reports when they are created, VERL's Torch traces when they are exported, and
the TokenSpeed scheduler profiles a rollout server stopped, under the server
actor that requested them with the scheduler's ``rank_tag``. The
record is the only source of ownership: an artifact belongs to the process
that registered it, with the step and role it registered. Nothing is inferred
from filenames, pids or ranks.

Processes are keyed by host, pid and run, so a pid reused by another run (e.g.
a repeated containerized job) is a different process.

Problems are reported instead of raised, so a partial run still merges:

* ``incomplete``: an artifact or record cut short, unreadable or malformed.
* ``duplicate``: a copy of an artifact already seen (same file name and
  content), or a second record for one process.
* ``missing``: an artifact a process registered but that is not in the inputs.
* ``unlinked``: an artifact no process registered, e.g. from a run without this
  package; it is merged on its own.

:func:`select` then groups artifacts by run and step: every process record
names its run, registered profiler artifacts name their step, and the
trainer's ``global_step`` spans give each step's time window. It keeps one run
and, optionally, some steps, and reports:

* ``mixed_runs``: the inputs hold artifacts of several runs and none was chosen.
* ``no_step_window``: a chosen step has no ``global_step`` span, so RL-Insight
  spans cannot be cut to it.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from rl_trace_observer.context import STEP_MARKER_ATTRIBUTE, STEP_SPAN_NAME, profile_session_id
from rl_trace_observer.process_record import SCHEMA_VERSION as RECORD_SCHEMA_VERSION

from .sources import PROCESS_RECORD, RL_INSIGHT, ArtifactError, TraceSource, discover, read_source

# 2: processes[].artifacts maps paths to record entries; artifacts[] carry no os_pid or rank.
SCHEMA_VERSION = 2


@dataclass
class Process:
    key: str
    host: str
    hostname: str
    os_pid: int
    rank: int | None = None
    actor_name: str | None = None
    ray: dict[str, Any] | None = None
    clock: dict[str, int] | None = None
    record: str | None = None
    run_id: str | None = None
    role: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    # Registered artifact path -> its record entry (kind, file, global_step, role, rank_tag).
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def title(self) -> str:
        title = f"{self.hostname} pid {self.os_pid}"
        if self.actor_name:
            title += f" · {self.actor_name}"
        elif self.rank is not None:
            title += f" · rank {self.rank}"
        return title


@dataclass
class Artifact:
    kind: str
    path: str
    size_bytes: int
    sha256: str
    process: str | None = None
    complete: bool = True
    run_id: str | None = None
    global_step: int | None = None
    profile_session_id: str | None = None
    role: str | None = None
    rank_tag: str | None = None
    # False when the artifact is outside the chosen run or steps.
    selected: bool = True


@dataclass
class ProfileSession:
    """One profiled step of a run: its time window and its step-specific artifacts."""

    id: str
    run_id: str | None
    global_step: int
    # From the trainer's global_step span; None when it was not recorded.
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    artifacts: list[str] = field(default_factory=list)


@dataclass
class Problem:
    kind: str
    path: str
    detail: str


@dataclass
class SessionManifest:
    inputs: list[str]
    processes: dict[str, Process] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    problems: list[Problem] = field(default_factory=list)
    runs: list[str] = field(default_factory=list)
    sessions: list[ProfileSession] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION
    # Problems added by the last select(), replaced by the next one.
    _selection_problems: list[Problem] = field(default_factory=list, repr=False)

    def to_json(self, **extra: Any) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inputs": self.inputs,
            **extra,
            "runs": self.runs,
            "sessions": [asdict(session) for session in self.sessions],
            "processes": [asdict(process) for process in self.processes.values()],
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "problems": [asdict(problem) for problem in self.problems],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Optional process-record fields and the JSON types they must have.
_RECORD_FIELDS: dict[str, type] = {
    "hostname": str,
    "run_id": str,
    "role": str,
    "ray": dict,
    "torch_distributed": dict,
    "clock": dict,
    "versions": dict,
    "artifacts": list,
}
_ENTRY_FIELDS: dict[str, type] = {"kind": str, "file": str, "global_step": int, "role": str, "rank_tag": str}


def _check_types(data: object, fields: dict[str, type], what: str) -> None:
    if not isinstance(data, dict):
        raise TypeError(f"{what} must be a JSON object")
    for name, expected in fields.items():
        value = data.get(name)
        if value is not None and (not isinstance(value, expected) or isinstance(value, bool)):
            raise TypeError(f"{what} field {name} has the wrong type: {value!r}")


def _read_record(path: Path) -> Process:
    """Read and validate a process record, so later steps can trust it."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        _check_types(record, _RECORD_FIELDS, "record")
        if record.get("schema_version") != RECORD_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {record.get('schema_version')!r}")
        entries = record.get("artifacts") or []
        for entry in entries:
            _check_types(entry, _ENTRY_FIELDS, "artifact entry")
            if "file" not in entry:
                raise KeyError("file")
        host, os_pid = str(record["host"]), int(record["os_pid"])
        run_id = record.get("run_id")
        ray = record.get("ray") or {}
        return Process(
            key=f"{host}:{os_pid}" if run_id is None else f"{host}:{os_pid}@{run_id}",
            host=host,
            hostname=record.get("hostname") or host,
            os_pid=os_pid,
            rank=(record.get("torch_distributed") or {}).get("rank"),
            actor_name=ray.get("actor_name"),
            ray=record.get("ray"),
            clock=record.get("clock"),
            record=str(path),
            run_id=run_id,
            role=record.get("role"),
            versions=dict(record.get("versions") or {}),
            # Entries are relative to the record, so a copied output directory still links.
            artifacts={str((path.parent / entry["file"]).resolve()): entry for entry in entries},
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ArtifactError(f"Cannot read process record {path}: {error!r}") from error


def build_manifest(inputs: Iterable[Path]) -> tuple[SessionManifest, list[TraceSource]]:
    """Discover, read and link every artifact under ``inputs``."""
    inputs = list(inputs)
    manifest = SessionManifest(inputs=[str(path) for path in inputs])
    found = discover(inputs)

    for kind, path in found:
        if kind != PROCESS_RECORD:
            continue
        try:
            process = _read_record(path)
        except ArtifactError as error:
            manifest.problems.append(Problem("incomplete", str(path), str(error)))
            continue
        if process.key in manifest.processes:
            manifest.problems.append(Problem("duplicate", str(path), f"second record for process {process.key}"))
            continue
        manifest.processes[process.key] = process
    registered_by = {path: process for process in manifest.processes.values() for path in process.artifacts}
    manifest.runs = sorted({process.run_id for process in manifest.processes.values() if process.run_id})

    sources: list[TraceSource] = []
    # Artifact names carry the writer's identity (host and pid, or pid and a
    # timestamp), so only a file with both the same name and content is a copy.
    # Empty files are exempt: idle processes of two runs that reused a host and
    # pid write identical empty JSONL files, and there is nothing to merge twice.
    seen: dict[tuple[str, str], str] = {}
    # Registered files first, so of a file and its copy the copy is the duplicate.
    for kind, path in sorted(found, key=lambda item: str(item[1]) not in registered_by):
        if kind == PROCESS_RECORD:
            continue
        artifact = Artifact(kind=kind, path=str(path), size_bytes=0, sha256="")
        manifest.artifacts.append(artifact)
        try:
            artifact.size_bytes, artifact.sha256 = path.stat().st_size, _sha256(path)
        except OSError as error:
            artifact.complete = False
            manifest.problems.append(Problem("incomplete", str(path), f"cannot read: {error}"))
            continue
        identity = (path.name, artifact.sha256)
        if identity in seen and artifact.size_bytes:
            artifact.complete = False
            manifest.problems.append(Problem("duplicate", str(path), f"copy of {seen[identity]}"))
            continue
        seen[identity] = str(path)
        try:
            source = read_source(kind, path)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            # ArtifactError, valid JSON with malformed fields (e.g. a null
            # line), or a file that became unreadable after hashing.
            artifact.complete = False
            manifest.problems.append(Problem("incomplete", str(path), str(error) or repr(error)))
            continue
        artifact.complete = source.complete
        if not source.complete:
            manifest.problems.append(Problem("incomplete", str(path), "final line truncated"))

        process = registered_by.get(str(path))
        if process is None:
            manifest.problems.append(Problem("unlinked", str(path), "no process record registered it"))
        else:
            entry = process.artifacts[str(path)]
            source.process, source.host = process.key, process.host
            source.global_step = entry.get("global_step")
            source.role = entry.get("role") or process.role
            source.rank_tag = entry.get("rank_tag") or source.rank_tag
            artifact.process, artifact.run_id = process.key, process.run_id
            artifact.global_step, artifact.role = source.global_step, source.role
            artifact.profile_session_id = profile_session_id(artifact.run_id, artifact.global_step)
        artifact.rank_tag = source.rank_tag
        sources.append(source)

    found_paths = {str(path) for _, path in found}
    for process in manifest.processes.values():
        for registered in process.artifacts:
            if registered not in found_paths:
                manifest.problems.append(Problem("missing", registered, f"registered by process {process.key}"))
    return manifest, sources


def _run_of(source: TraceSource, processes: dict[str, Process]) -> str | None:
    process = processes.get(source.process) if source.process is not None else None
    return process.run_id if process is not None else None


def _step_windows(
    sources: list[TraceSource], processes: dict[str, Process], problems: list[Problem]
) -> dict[tuple, tuple[int, int]]:
    """Map ``(run_id, global_step)`` to the step's absolute time window in epoch ns."""
    windows: dict[tuple, tuple[int, int]] = {}
    for source in sources:
        if source.kind != RL_INSIGHT:
            continue
        malformed = 0
        for event in source.events:
            args = event.get("args")
            if event.get("name") != STEP_SPAN_NAME or not isinstance(args, dict) or not args.get(STEP_MARKER_ATTRIBUTE):
                continue
            try:
                marker_run = args.get("run_id")
                if marker_run is not None and not isinstance(marker_run, str):
                    raise TypeError(f"run_id must be a string, got {marker_run!r}")
                start = source.base_ns + round(float(event["ts"]) * 1000)
                end = start + round(float(event.get("dur", 0)) * 1000)
                key = (marker_run or _run_of(source, processes), int(args["global_step"]))
            except (KeyError, TypeError, ValueError, OverflowError):
                malformed += 1
                continue
            previous = windows.get(key)
            windows[key] = (min(start, previous[0]), max(end, previous[1])) if previous else (start, end)
        if malformed:
            problems.append(Problem("incomplete", str(source.path), f"{malformed} malformed {STEP_SPAN_NAME} spans"))
    return windows


def _overlaps(source: TraceSource, event: dict[str, Any], windows: list[tuple[int, int]]) -> bool:
    start = source.base_ns + round(float(event["ts"]) * 1000)
    end = start + round(float(event.get("dur", 0)) * 1000)
    return any(start <= window_end and end >= window_start for window_start, window_end in windows)


def _sessions(windows: dict[tuple, tuple[int, int]], artifacts: list[Artifact]) -> list[ProfileSession]:
    sessions: dict[tuple, ProfileSession] = {}
    for (run_id, step), (start, end) in windows.items():
        sessions[(run_id, step)] = ProfileSession(
            id=profile_session_id(run_id, step) or f"step-{step}",
            run_id=run_id,
            global_step=step,
            start_time_ns=start,
            end_time_ns=end,
        )
    for artifact in artifacts:
        if artifact.global_step is None:
            continue
        session = sessions.setdefault(
            (artifact.run_id, artifact.global_step),
            ProfileSession(
                id=artifact.profile_session_id or f"step-{artifact.global_step}",
                run_id=artifact.run_id,
                global_step=artifact.global_step,
            ),
        )
        session.artifacts.append(artifact.path)
    return sorted(sessions.values(), key=lambda session: (session.run_id or "", session.global_step))


def _cut_to_windows(source: TraceSource, windows: list[tuple[int, int]], problems: list[Problem]) -> TraceSource:
    """A copy of ``source`` with the events that overlap ``windows``; the original stays whole."""
    events, malformed = [], 0
    for event in source.events:
        if event.get("ph") == "M" or "ts" not in event:
            events.append(event)
            continue
        try:
            if _overlaps(source, event, windows):
                events.append(event)
        except (TypeError, ValueError, OverflowError):
            malformed += 1
    if malformed:
        problems.append(Problem("incomplete", str(source.path), f"{malformed} events with a malformed time"))
    return replace(source, events=events)


def select(
    manifest: SessionManifest,
    sources: list[TraceSource],
    *,
    run: str | None = None,
    steps: Iterable[int] | None = None,
) -> list[TraceSource]:
    """Record the manifest's profile sessions and keep the sources of ``run`` and ``steps``.

    Torch and VizTracer sources are kept when their registered step is chosen;
    RL-Insight sources keep the events that overlap a chosen step's window.
    Without ``steps`` every step is kept; without ``run`` the inputs must hold
    one run. Repeated calls start from the same ``sources``.
    """
    steps = sorted(set(steps)) if steps is not None else None
    processes = manifest.processes
    for problem in manifest._selection_problems:
        manifest.problems.remove(problem)
    problems: list[Problem] = []
    windows = _step_windows(sources, processes, problems)
    manifest.sessions = _sessions(windows, manifest.artifacts)

    inputs = ", ".join(manifest.inputs)
    if run is None and len(manifest.runs) > 1:
        problems.append(Problem("mixed_runs", inputs, f"artifacts of runs {', '.join(manifest.runs)}"))

    artifacts = {artifact.path: artifact for artifact in manifest.artifacts}
    for artifact in manifest.artifacts:
        # Unreadable artifacts are never merged; readable ones are re-marked below.
        artifact.selected = False
    kept = []
    for source in sources:
        run_id = _run_of(source, processes)
        keep = run is None or run_id == run
        if keep and steps is not None:
            if source.kind == RL_INSIGHT:
                # Windows of this source's run; a source or span of unknown run matches any.
                chosen = [
                    window
                    for (window_run, step), window in windows.items()
                    if step in steps and (run_id is None or window_run is None or window_run == run_id)
                ]
                source = _cut_to_windows(source, chosen, problems)
                keep = bool(chosen)
            else:
                keep = source.global_step in steps
        artifacts[str(source.path)].selected = keep
        if keep:
            kept.append(source)

    # A chosen step needs a window of the chosen run, whatever sources there are.
    for step in steps or []:
        if not any(key[1] == step and (run is None or key[0] in (run, None)) for key in windows):
            detail = f"no {STEP_SPAN_NAME} span for step {step}; RL-Insight spans left out"
            problems.append(Problem("no_step_window", inputs, detail))
    manifest.problems.extend(problems)
    manifest._selection_problems = problems
    return kept
