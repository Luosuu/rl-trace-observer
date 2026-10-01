"""Per-process artifact records written next to the trace artifacts.

Every process that writes an artifact also writes one record,
``rl-trace-process-<host>-pid-<pid>.json``, describing who it is and which files
it wrote or requested from a server it drives (TokenSpeed scheduler profiles).
The merger joins records with artifacts to build the session manifest:
records give Torch and VizTracer traces, whose filenames carry only a pid, the
host and Ray identity of the process that wrote them, and let it report
artifacts that a process registered but that never arrived.
"""

import json
import os
import socket
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from rl_trace_observer.context import current_run_id
from rl_trace_observer.output import safe_component

SCHEMA_VERSION = 1
# Distributions whose version is recorded once their module is imported.
_VERSIONED_MODULES = {
    "rl_trace_observer": "rl-trace-observer",
    "rl_insight": "rl-insight",
    "verl": "verl",
    "torch": "torch",
    "viztracer": "viztracer",
    "tokenspeed": "tokenspeed",
    "ray": "ray",
}
RECORD_PREFIX = "rl-trace-process-"

_lock = threading.Lock()
_records: dict[Path, dict[str, Any]] = {}


def record_path(output_dir: Path) -> Path:
    return output_dir / f"{RECORD_PREFIX}{safe_component(socket.gethostname())}-pid-{os.getpid()}.json"


def _ray_identity() -> dict[str, Any] | None:
    ray = sys.modules.get("ray")
    if ray is None or not ray.is_initialized():
        return None
    context = ray.get_runtime_context()
    identity = {}
    for key, getter in (
        ("job_id", context.get_job_id),
        ("node_id", context.get_node_id),
        ("worker_id", context.get_worker_id),
        ("actor_id", context.get_actor_id),
        ("actor_name", context.get_actor_name),
    ):
        try:
            # Unnamed actors report an empty name; drivers have no actor at all.
            identity[key] = getter() or None
        except Exception:
            identity[key] = None
    return identity


def _torch_distributed() -> dict[str, int] | None:
    # Only inspect torch when the process already imported it.
    distributed = sys.modules.get("torch.distributed")
    if distributed is None or not distributed.is_available() or not distributed.is_initialized():
        return None
    return {"rank": distributed.get_rank(), "world_size": distributed.get_world_size()}


def _versions() -> dict[str, str]:
    versions = {}
    for module, distribution in _VERSIONED_MODULES.items():
        if module in sys.modules:
            try:
                versions[distribution] = metadata.version(distribution)
            except metadata.PackageNotFoundError:
                pass
    return versions


def _update(output_dir: Path, entry: dict[str, Any] | None = None, process_role: str | None = None) -> Path:
    """Refresh this process's record, add ``entry``, and rewrite the record atomically.

    The process identity (run id, Ray, ``torch.distributed``, versions) is
    re-read on every call, so a record first written before Ray or the process
    group was initialized picks them up later.
    """
    # The same directory can be reached through a symlink or "..": key the
    # cache, and the relative artifact paths, by the resolved directory.
    output_dir = output_dir.resolve()
    target = record_path(output_dir)
    with _lock:
        record = _records.setdefault(
            target,
            {
                "schema_version": SCHEMA_VERSION,
                "hostname": socket.gethostname(),
                "host": safe_component(socket.gethostname()),
                "os_pid": os.getpid(),
                "clock": {"wall_time_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()},
                "role": None,
                "artifacts": [],
            },
        )
        record["run_id"] = current_run_id()
        record["role"] = record["role"] or process_role
        record["ray"] = _ray_identity()
        record["torch_distributed"] = _torch_distributed()
        record["versions"] = _versions()
        if entry is not None:
            entry = {**entry, "file": os.path.relpath(Path(entry["file"]).resolve(), output_dir)}
            if entry not in record["artifacts"]:
                record["artifacts"].append(entry)
        record["updated_wall_time_ns"] = time.time_ns()

        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)
    return target


def register_artifact(
    output_dir: Path,
    kind: str,
    path: Path,
    *,
    global_step: int | None = None,
    role: str | None = None,
    rank_tag: str | None = None,
) -> Path:
    """Record that this process owns ``path``; the merger links the artifact through it.

    Args:
        global_step: The training step the artifact covers, when it covers one.
        role: What the artifact records, e.g. ``"actor_train"``.
        rank_tag: The TokenSpeed scheduler rank that wrote the artifact, e.g.
            ``"TP0"``, when this process requested it from a TokenSpeed server.
    """
    entry = {"kind": kind, "file": str(path), "global_step": global_step, "role": role, "rank_tag": rank_tag}
    return _update(output_dir, {key: value for key, value in entry.items() if value is not None})


def set_process_role(output_dir: Path, role: str) -> Path:
    """Record this process's role, e.g. ``"trainer"``; the first role set is kept."""
    return _update(output_dir, process_role=role)
