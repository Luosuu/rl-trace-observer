import os

import pytest

# Under `uv run`, Ray otherwise rebuilds a separate uv environment for every
# worker from the bare project, without the extras synced for this run. Test
# workers must use the same environment as the driver. Ray reads this on import.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

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
