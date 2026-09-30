"""Build the session manifest: which artifacts exist, whose they are, what is wrong.

A session is the set of artifacts under the merger's inputs. Process records
(``rl-trace-process-<host>-pid-<pid>.json``) identify each OS process that wrote
artifacts. The manifest links every artifact to a process:

* RL-Insight JSONL names its host and pid in the filename.
* Torch and VizTracer filenames carry only a pid (and a rank), so they are
  matched to the record with that pid, using the rank when several hosts
  reused the same pid.

Problems are reported instead of raised, so a partial run still merges:

* ``incomplete``: an artifact cut short or unreadable.
* ``duplicate``: a copy of an artifact already seen (same file name and content).
* ``missing``: an artifact a process registered but that is not in the inputs.
* ``unlinked`` / ``ambiguous``: no process record, or several, match an artifact.
"""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

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
    schema_version: int = SCHEMA_VERSION

    def to_json(self, **extra: Any) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inputs": self.inputs,
            **extra,
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
        return Process(
            key=f"{host}:{os_pid}",
            host=host,
            hostname=record.get("hostname", host),
            os_pid=os_pid,
            rank=distributed.get("rank"),
            actor_name=(ray or {}).get("actor_name"),
            ray=ray,
            clock=record.get("clock"),
            record=str(path),
            artifacts=[str(path.parent / entry["file"]) for entry in record.get("artifacts", [])],
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ArtifactError(f"Cannot read process record {path}: {error!r}") from error


def _link(source: TraceSource, processes: dict[str, Process]) -> tuple[str | None, Problem | None]:
    """Return the key of the process that wrote ``source``, or the problem why not."""
    if source.process is not None:
        return source.process, None
    # A process whose known rank differs from the source's cannot have written
    # it, even when it is the only one with that pid (another host's record may
    # be missing).
    candidates = [
        process
        for process in processes.values()
        if process.os_pid == source.os_pid
        and (source.rank is None or process.rank is None or process.rank == source.rank)
    ]
    if len(candidates) > 1 and source.rank is not None:
        candidates = [process for process in candidates if process.rank == source.rank]
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

    sources: list[TraceSource] = []
    # Artifact names carry the writer's identity (host and pid, or pid and a
    # timestamp), so only a file with both the same name and content is a copy;
    # e.g. every process without spans writes an identical empty JSONL.
    seen: dict[tuple[str, str], str] = {}
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
        if identity in seen:
            artifact.complete = False
            manifest.problems.append(Problem("duplicate", str(path), f"copy of {seen[identity]}"))
            continue
        seen[identity] = str(path)
        try:
            source = read_source(kind, path)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            # ArtifactError, or valid JSON with malformed fields (e.g. a null line).
            artifact.complete = False
            manifest.problems.append(Problem("incomplete", str(path), str(error) or repr(error)))
            continue

        artifact.os_pid, artifact.rank, artifact.complete = source.os_pid, source.rank, source.complete
        if not source.complete:
            manifest.problems.append(Problem("incomplete", str(path), "final line truncated"))
        source.process, problem = _link(source, manifest.processes)
        if problem is not None:
            manifest.problems.append(problem)
        elif source.kind == RL_INSIGHT and source.process not in manifest.processes:
            # Older runs wrote no process records; the filename still names the process.
            manifest.processes[source.process] = Process(
                key=source.process, host=source.host, hostname=source.host, os_pid=source.os_pid
            )
        if source.process in manifest.processes:
            source.host = manifest.processes[source.process].host
        artifact.process = source.process
        sources.append(source)

    found_paths = {str(path) for _, path in found}
    for process in manifest.processes.values():
        for registered in process.artifacts:
            if str(Path(registered).resolve()) not in found_paths:
                manifest.problems.append(Problem("missing", registered, f"registered by process {process.key}"))
    return manifest, sources
