import os

import pytest

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
