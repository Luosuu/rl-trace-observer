"""Build the session manifest: which artifacts exist, whose they are, what is wrong.

A session is the set of artifacts under the merger's inputs. Process records
(``rl-trace-process-<host>-pid-<pid>.json``) identify each OS process that wrote
artifacts. The manifest links every artifact to a process:

* An artifact a process registered is linked to that process's record.
* Otherwise RL-Insight JSONL is matched by the host and pid in its filename.
* Torch and VizTracer filenames carry only a pid (and a rank), so they are
  matched to the record with that pid, using the rank when several hosts
  reused the same pid.

Processes are keyed by host, pid and run, so a pid reused by another run (e.g.
a repeated containerized job) is a different process.

Problems are reported instead of raised, so a partial run still merges:

* ``incomplete``: an artifact cut short or unreadable.
* ``duplicate``: a copy of an artifact already seen (same file name and content).
* ``missing``: an artifact a process registered but that is not in the inputs.
* ``unlinked`` / ``ambiguous``: no process record, or several, match an artifact.

:func:`select` then groups artifacts by :mod:`rl_trace_observer.context`: every
process record names its run, profiler windows name their step, and the
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

SCHEMA_VERSION = 1


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
    artifacts: list[str] = field(default_factory=list)
    run_id: str | None = None
    role: str | None = None
    versions: dict[str, str] = field(default_factory=dict)
    # Last time the process updated its record (epoch ns).
    updated_wall_time_ns: int | None = None
    # Registered artifact path -> the global step the process said it covers.
    artifact_steps: dict[str, int] = field(default_factory=dict)

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
    os_pid: int | None = None
    rank: int | None = None
    process: str | None = None
    complete: bool = True
    run_id: str | None = None
    global_step: int | None = None
    profile_session_id: str | None = None
    role: str | None = None
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


def _read_record(path: Path) -> Process:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("schema_version") != RECORD_SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {record.get('schema_version')!r}")
        host, os_pid = str(record["host"]), int(record["os_pid"])
        distributed = record.get("torch_distributed") or {}
        ray = record.get("ray")
        run_id, role = record.get("run_id"), record.get("role")
        for name, value in (("run_id", run_id), ("role", role)):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string, got {value!r}")
        return Process(
            key=f"{host}:{os_pid}" if run_id is None else f"{host}:{os_pid}@{run_id}",
            host=host,
            hostname=record.get("hostname", host),
            os_pid=os_pid,
            rank=distributed.get("rank"),
            actor_name=(ray or {}).get("actor_name"),
            ray=ray,
            clock=record.get("clock"),
            record=str(path),
            artifacts=[str((path.parent / entry["file"]).resolve()) for entry in record.get("artifacts", [])],
            run_id=run_id,
            role=role,
            updated_wall_time_ns=record.get("updated_wall_time_ns"),
            artifact_steps={
                str((path.parent / entry["file"]).resolve()): int(entry["global_step"])
                for entry in record.get("artifacts", [])
                if entry.get("global_step") is not None
            },
            versions=dict(record.get("versions") or {}),
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ArtifactError(f"Cannot read process record {path}: {error!r}") from error


def _run_spans(processes: dict[str, Process]) -> dict[str, tuple[int, int]]:
    """Time each run's records cover: its first record to its last update (epoch ns).

    The trainer updates its record after every step, so a run's span ends after
    its last profiled step starts.
    """
    spans: dict[str, tuple[int, int]] = {}
    for process in processes.values():
        start = (process.clock or {}).get("wall_time_ns")
        end = process.updated_wall_time_ns
        if process.run_id is None or not isinstance(start, int) or not isinstance(end, int):
            continue
        previous = spans.get(process.run_id, (start, end))
        spans[process.run_id] = (min(start, previous[0]), max(end, previous[1]))
    return spans


def _start_ns(source: TraceSource) -> int | None:
    starts = [float(event["ts"]) for event in source.events if event.get("ph") != "M" and "ts" in event]
    return source.base_ns + round(min(starts) * 1000) if starts else None


def _link(
    source: TraceSource,
    processes: dict[str, Process],
    registered_by: dict[str, str],
    run_spans: dict[str, tuple[int, int]] | None = None,
) -> tuple[str | None, Problem | None]:
    """Return the key of the process that wrote ``source``, or the problem why not.

    An RL-Insight source that no record matches returns ``(None, None)``: its
    filename still names its process.
    """
    if (key := registered_by.get(str(source.path))) is not None:
        return key, None
    if source.kind == RL_INSIGHT:
        candidates = [
            process for process in processes.values() if process.host == source.host and process.os_pid == source.os_pid
        ]
        if len(candidates) > 1:
            runs = ", ".join(sorted(str(process.run_id) for process in candidates))
            return None, Problem("ambiguous", str(source.path), f"{source.host} pid {source.os_pid} in runs {runs}")
        return (candidates[0].key if candidates else None), None
    # A process whose known rank differs from the source's cannot have written
    # it, even when it is the only one with that pid (another host's record may
    # be missing).
    candidates = [
        process
        for process in processes.values()
        if process.os_pid == source.os_pid
        and (source.rank is None or process.rank is None or process.rank == source.rank)
    ]
    # Prefer the exact rank; records written before torch.distributed was
    # initialized have none, and several of those stay ambiguous.
    exact = [process for process in candidates if source.rank is not None and process.rank == source.rank]
    if len(candidates) > 1 and exact:
        candidates = exact
    # Runs that reused the pid (and rank): keep the run whose records span the
    # trace. VERL's Torch traces are not registered by any process.
    if len(candidates) > 1 and run_spans and (start := _start_ns(source)) is not None:
        during = [
            process
            for process in candidates
            if process.run_id in run_spans and run_spans[process.run_id][0] <= start <= run_spans[process.run_id][1]
        ]
        if during:
            candidates = during
    if len(candidates) == 1:
        return candidates[0].key, None
    if not candidates:
        rank = f" and rank {source.rank}" if source.rank is not None else ""
        return None, Problem("unlinked", str(source.path), f"no process record for pid {source.os_pid}{rank}")
    hosts = ", ".join(sorted(process.host for process in candidates))
    return None, Problem("ambiguous", str(source.path), f"pid {source.os_pid} matches processes on {hosts}")


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
    registered_by = {path: process.key for process in manifest.processes.values() for path in process.artifacts}
    run_spans = _run_spans(manifest.processes)

    sources: list[TraceSource] = []
    # Artifact names carry the writer's identity (host and pid, or pid and a
    # timestamp), so only a file with both the same name and content is a copy;
    # e.g. every process without spans writes an identical empty JSONL.
    seen: dict[tuple[str, str], str] = {}
    fallback: dict[str, Process] = {}
    for kind, path in found:
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
        # Empty files hold nothing to merge twice, and idle processes of two
        # runs that reused a host and pid write identical empty files.
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

        artifact.os_pid, artifact.rank, artifact.complete = source.os_pid, source.rank, source.complete
        artifact.global_step, artifact.role = source.global_step, source.role
        if not source.complete:
            manifest.problems.append(Problem("incomplete", str(path), "final line truncated"))
        source.process, problem = _link(source, manifest.processes, registered_by, run_spans)
        if problem is not None:
            manifest.problems.append(problem)
        elif source.kind == RL_INSIGHT and source.process is None:
            # Older runs wrote no process records; the filename still names the
            # process. Such a process only names the RL-Insight source: it is
            # kept out of linking, where its pid could claim another host's trace.
            source.process = f"{source.host}:{source.os_pid}"
            fallback.setdefault(
                source.process,
                Process(key=source.process, host=source.host, hostname=source.host, os_pid=source.os_pid),
            )
        if source.process in manifest.processes:
            source.host = manifest.processes[source.process].host
        artifact.process = source.process
        sources.append(source)

    manifest.processes.update(fallback)
    readable = {str(source.path): source for source in sources}
    for artifact in manifest.artifacts:
        if (process := manifest.processes.get(artifact.process)) is not None:
            artifact.run_id = process.run_id
            # The filename's role and step win; the record fills in what it lacks.
            artifact.role = artifact.role or process.role
            if artifact.global_step is None and artifact.path in process.artifact_steps:
                artifact.global_step = process.artifact_steps[artifact.path]
                readable[artifact.path].global_step = artifact.global_step
        artifact.profile_session_id = profile_session_id(artifact.run_id, artifact.global_step)
    manifest.runs = sorted({process.run_id for process in manifest.processes.values() if process.run_id})
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
    sources: list[TraceSource],
    processes: dict[str, Process],
    problems: list[Problem],
    marker_runs: dict[str, set[str]],
) -> dict[tuple, tuple[int, int]]:
    """Map ``(run_id, global_step)`` to the step's absolute time window in epoch ns.

    ``marker_runs`` receives, per source path, the run ids its markers name.
    """
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
                start = source.base_ns + round(float(event["ts"]) * 1000)
                end = start + round(float(event.get("dur", 0)) * 1000)
                marker_run = args.get("run_id")
                if marker_run is not None and not isinstance(marker_run, str):
                    raise TypeError(f"run_id must be a string, got {marker_run!r}")
                key = (marker_run or _run_of(source, processes), int(args["global_step"]))
            except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
                malformed += 1
                continue
            if key[0] is not None:
                marker_runs.setdefault(str(source.path), set()).add(key[0])
            previous = windows.get(key)
            windows[key] = (min(start, previous[0]), max(end, previous[1])) if previous else (start, end)
        if malformed:
            problems.append(Problem("incomplete", str(source.path), f"{malformed} malformed {STEP_SPAN_NAME} spans"))
    return windows


def _overlaps(source: TraceSource, event: dict[str, Any], windows: list[tuple[int, int]]) -> bool:
    start = source.base_ns + round(float(event["ts"]) * 1000)
    end = start + round(float(event.get("dur", 0)) * 1000)
    return any(start <= window_end and end >= window_start for window_start, window_end in windows)


def select(
    manifest: SessionManifest,
    sources: list[TraceSource],
    *,
    run: str | None = None,
    steps: Iterable[int] | None = None,
) -> list[TraceSource]:
    """Record the manifest's profile sessions and keep the sources of ``run`` and ``steps``.

    Torch and VizTracer sources are kept when their step is chosen; RL-Insight
    sources keep the events that overlap a chosen step's window. Without
    ``steps`` every step is kept; without ``run`` the inputs must hold one run.
    """
    steps = sorted(set(steps)) if steps is not None else None
    processes = manifest.processes
    for problem in manifest._selection_problems:
        manifest.problems.remove(problem)
    problems: list[Problem] = []
    marker_runs: dict[str, set[str]] = {}
    windows = _step_windows(sources, processes, problems, marker_runs)

    # A source without a process record takes the one run its step markers name.
    artifacts = {artifact.path: artifact for artifact in manifest.artifacts}
    source_runs = {}
    for source in sources:
        run_id = _run_of(source, processes)
        if run_id is None and len(named := marker_runs.get(str(source.path), set())) == 1:
            [run_id] = named
        source_runs[str(source.path)] = run_id
        artifact = artifacts[str(source.path)]
        artifact.run_id = run_id
        artifact.profile_session_id = profile_session_id(run_id, artifact.global_step)
    manifest.runs = sorted(
        {process.run_id for process in processes.values() if process.run_id}
        | {artifact.run_id for artifact in manifest.artifacts if artifact.run_id}
    )

    sessions: dict[tuple, ProfileSession] = {}
    for (run_id, step), (start, end) in windows.items():
        sessions[(run_id, step)] = ProfileSession(
            id=profile_session_id(run_id, step) or f"step-{step}",
            run_id=run_id,
            global_step=step,
            start_time_ns=start,
            end_time_ns=end,
        )
    for artifact in manifest.artifacts:
        if artifact.global_step is None:
            continue
        key = (artifact.run_id, artifact.global_step)
        session = sessions.setdefault(
            key,
            ProfileSession(
                id=artifact.profile_session_id or f"step-{artifact.global_step}",
                run_id=artifact.run_id,
                global_step=artifact.global_step,
            ),
        )
        session.artifacts.append(artifact.path)
    manifest.sessions = sorted(sessions.values(), key=lambda session: (session.run_id or "", session.global_step))

    inputs = ", ".join(manifest.inputs)
    if run is None and len(manifest.runs) > 1:
        problems.append(Problem("mixed_runs", inputs, f"artifacts of runs {', '.join(manifest.runs)}"))

    for artifact in manifest.artifacts:
        # Unreadable artifacts are never merged; readable ones are re-marked below.
        artifact.selected = False
    kept = []
    for source in sources:
        run_id = source_runs[str(source.path)]
        keep = run is None or run_id == run
        if keep and steps is not None:
            if source.kind == RL_INSIGHT:
                # Windows of this source's run; a source or span of unknown run matches any.
                chosen = [
                    window
                    for (window_run, step), window in windows.items()
                    if step in steps and (run_id is None or window_run is None or window_run == run_id)
                ]
                events, malformed = [], 0
                for event in source.events:
                    if event.get("ph") == "M" or "ts" not in event:
                        events.append(event)
                        continue
                    try:
                        if _overlaps(source, event, chosen):
                            events.append(event)
                    except (TypeError, ValueError, OverflowError):
                        malformed += 1
                if malformed:
                    problems.append(
                        Problem("incomplete", str(source.path), f"{malformed} events with a malformed time")
                    )
                # A copy: the caller's sources stay whole for another selection.
                source = replace(source, events=events)
                keep = bool(chosen)
            else:
                keep = source.global_step in steps
        artifacts[str(source.path)].selected = keep
        if keep:
            kept.append(source)
    # A chosen step needs a window of the chosen run, whatever sources there are.
    missing_windows = [
        step
        for step in steps or []
        if not any(key[1] == step and (run is None or key[0] in (run, None)) for key in windows)
    ]
    for step in missing_windows:
        problems.append(
            Problem("no_step_window", inputs, f"no {STEP_SPAN_NAME} span for step {step}; RL-Insight spans left out")
        )
    manifest.problems.extend(problems)
    manifest._selection_problems = problems
    return kept
