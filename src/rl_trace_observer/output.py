import os
import re
from pathlib import Path

OUTPUT_DIR_ENV = "RL_TRACE_OUTPUT_DIR"


def trace_output_dir() -> Path:
    """Return the artifact directory from ``RL_TRACE_OUTPUT_DIR``, creating it.

    The directory must be absolute: Ray workers started from a runtime env
    ``working_dir`` run inside a temporary copy of it, so a relative path would
    silently write artifacts that disappear with the Ray session.

    Raises:
        ValueError: ``RL_TRACE_OUTPUT_DIR`` is unset or relative.
    """
    value = os.getenv(OUTPUT_DIR_ENV, "").strip()
    if not value:
        raise ValueError(f"{OUTPUT_DIR_ENV} must be set to an absolute directory shared by all nodes")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"{OUTPUT_DIR_ENV} must be an absolute path, got {value!r}; relative paths resolve inside "
            "each Ray worker's temporary working_dir"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_component(value: object) -> str:
    """Make an arbitrary label safe to embed in an artifact filename."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"
