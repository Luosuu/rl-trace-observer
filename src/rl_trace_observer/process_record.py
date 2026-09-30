"""Per-process artifact records written next to the trace artifacts.

Every process that writes an artifact also writes one record,
``rl-trace-process-<host>-pid-<pid>.json``, describing who it is and which files
it wrote. The merger joins records with artifacts to build the session manifest:
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
from pathlib import Path
from typing import Any

from rl_trace_observer.output import safe_component

SCHEMA_VERSION = 1
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


def register_artifact(output_dir: Path, kind: str, path: Path) -> Path:
    """Add ``path`` to this process's record and rewrite the record atomically.

    The process identity is re-read on every call, so a record first written
    before ``torch.distributed`` or Ray was initialized picks them up later.
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
                "artifacts": [],
            },
        )
        record["ray"] = _ray_identity()
        record["torch_distributed"] = _torch_distributed()
        entry = {"kind": kind, "file": os.path.relpath(Path(path).resolve(), output_dir)}
        if entry not in record["artifacts"]:
            record["artifacts"].append(entry)
        record["updated_wall_time_ns"] = time.time_ns()

        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)
    return target
