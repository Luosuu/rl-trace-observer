"""TraceContext: which run, profile session and process an artifact belongs to.

A context is plain data that serializes to JSON and is rebuilt the same way in
every process, so artifacts are grouped by what the run says about itself rather
than by the directory they happen to be in:

* ``run_id``: ``RL_TRACE_RUN_ID`` when set, otherwise the Ray session and job
  (``<session_name>-job-<job_id>``), which every process of one training job
  shares without any propagation.
* ``profile_session_id``: one profiled step of a run, ``<run_id>-step-<n>``.
* ``role``, ``global_step``, ranks and ``request_id``: filled in by whoever
  knows them; ``None`` means unknown.
"""

import os
import socket
import sys
from dataclasses import asdict, dataclass, fields
from typing import Any

SCHEMA_VERSION = 1
RUN_ID_ENV = "RL_TRACE_RUN_ID"
# RL-Insight span the trainer records around each training step.
STEP_SPAN_NAME = "global_step"


def current_run_id() -> str | None:
    """Return this process's run id, or ``None`` outside a run."""
    if run_id := os.getenv(RUN_ID_ENV, "").strip():
        return run_id
    # Only inspect Ray when the process already uses it.
    ray = sys.modules.get("ray")
    if ray is None or not ray.is_initialized():
        return None
    context = ray.get_runtime_context()
    try:
        # A job id alone repeats across Ray clusters (every cluster starts at
        # 01000000); the session name identifies the cluster.
        return f"{context.get_session_name()}-job-{context.get_job_id()}"
    except Exception:
        return None


def profile_session_id(run_id: str | None, global_step: int | None) -> str | None:
    if run_id is None or global_step is None:
        return None
    return f"{run_id}-step-{global_step}"


@dataclass(frozen=True)
class TraceContext:
    run_id: str | None = None
    global_step: int | None = None
    role: str | None = None
    hostname: str | None = None
    os_pid: int | None = None
    framework_rank: int | None = None
    replica_rank: int | None = None
    request_id: str | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def profile_session_id(self) -> str | None:
        return profile_session_id(self.run_id, self.global_step)

    @classmethod
    def current(cls, **values: Any) -> "TraceContext":
        """Context of this process; ``values`` fill in or override what the caller knows,
        e.g. a ``run_id`` propagated to a process outside Ray."""
        defaults = {"run_id": current_run_id(), "hostname": socket.gethostname(), "os_pid": os.getpid()}
        return cls(**{**defaults, **values})

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["profile_session_id"] = self.profile_session_id
        return data

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TraceContext":
        """Rebuild a context; raises ``ValueError`` for another schema version."""
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported TraceContext schema_version {data.get('schema_version')!r}")
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})
