import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Under `uv run`, Ray's uv hook would replace the py_executable from
# ray_runtime_env.yaml with the driver's bare `uv run` flags. Tests exercise the
# yaml as written instead. Ray reads this on import.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")


def load_ray_runtime_env(**env_vars: str) -> dict:
    """Load the project's ray_runtime_env.yaml, as `ray job submit` would."""
    import yaml

    runtime_env = yaml.safe_load((REPO_ROOT / "ray_runtime_env.yaml").read_text())
    runtime_env["working_dir"] = str(REPO_ROOT / runtime_env["working_dir"])
    runtime_env["env_vars"] = {**runtime_env.get("env_vars", {}), **env_vars}
    return runtime_env


_BACKEND_ENV = "RL_INSIGHT_SERVER_BACKEND"


@pytest.fixture(autouse=True)
def _restore_rl_insight_state():
    """Keep RL-Insight registry and backend env changes from leaking between tests."""
    backend_env = os.environ.get(_BACKEND_ENV)
    try:
        from rl_insight.client.base import MONITOR_CLIENT_REGISTRY
    except ImportError:
        MONITOR_CLIENT_REGISTRY = None
    snapshot = dict(MONITOR_CLIENT_REGISTRY or {})

    yield

    if MONITOR_CLIENT_REGISTRY is not None:
        MONITOR_CLIENT_REGISTRY.clear()
        MONITOR_CLIENT_REGISTRY.update(snapshot)
    if backend_env is None:
        os.environ.pop(_BACKEND_ENV, None)
    else:
        os.environ[_BACKEND_ENV] = backend_env
