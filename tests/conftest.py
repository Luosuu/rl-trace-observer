import pytest


@pytest.fixture(autouse=True)
def _restore_rl_insight_registry():
    """Keep real RL-Insight registry changes from leaking between tests."""
    try:
        from rl_insight.client.base import MONITOR_CLIENT_REGISTRY
    except ImportError:
        yield
        return

    snapshot = dict(MONITOR_CLIENT_REGISTRY)
    yield
    MONITOR_CLIENT_REGISTRY.clear()
    MONITOR_CLIENT_REGISTRY.update(snapshot)
