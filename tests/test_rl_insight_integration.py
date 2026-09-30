"""Contract tests against the real RL-Insight package."""

import json

import pytest

rl_insight = pytest.importorskip("rl_insight")
pytest.importorskip("rl_insight.client.base")

from rl_trace_observer.integrations.rl_insight import register_rl_insight_client  # noqa: E402


@pytest.fixture
def monitor_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", "local://rl-trace-observer")
    monkeypatch.delenv("RL_TRACE_FORWARD_TO_RL_INSIGHT", raising=False)
    monkeypatch.delenv("RL_TRACE_CAPTURE_DEFAULT_BACKEND", raising=False)
    yield tmp_path
    rl_insight.finish()


def _read_events(output_dir):
    return [json.loads(line) for path in output_dir.glob("*.chrome.jsonl") for line in path.read_text().splitlines()]


def test_lazy_init_without_config_is_captured(monitor_env):
    """VERL workers call ``rl_insight.init()`` without the trainer config."""
    assert register_rl_insight_client()

    rl_insight.init()
    with rl_insight.trace_state("actor_update", state_lane_id="rank_0"):
        pass

    events = _read_events(monitor_env)
    assert [event["name"] for event in events] == ["actor_update"]
    assert events[0]["tid"] == "rank_0"


def test_explicit_backend_is_captured(monitor_env):
    assert register_rl_insight_client()

    rl_insight.init(config={"server": {"backend": "rl_trace_observer"}})
    with rl_insight.trace_state("train_batch"):
        pass

    assert [event["name"] for event in _read_events(monitor_env)] == ["train_batch"]


def test_rl_insight_without_backend_env_fails_fast(monkeypatch):
    from rl_insight.utils.constants import MonitorEnv

    monkeypatch.delattr(MonitorEnv, "SERVER_BACKEND")

    with pytest.raises(RuntimeError, match="RL_INSIGHT_SERVER_BACKEND"):
        register_rl_insight_client()
