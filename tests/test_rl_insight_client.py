import json
import sys
from unittest.mock import Mock

from rl_trace_observer.integrations.rl_insight.client import (
    BACKEND_NAME,
    ChromeTraceJsonlClient,
    create_rl_trace_observer_client,
    register_rl_insight_client,
)


def _trace_event():
    return {
        "kind": "trace",
        "name": "actor_update",
        "start_time_ns": 1_500_000,
        "end_time_ns": 3_750_000,
        "attributes": {
            "process_id": "1234",
            "state_lane_id": "rank_2",
            "state_name": "actor_update",
            "monitor.trace_segment": "state_interval",
        },
    }


def test_trace_event_is_written_as_chrome_jsonl(tmp_path):
    client = ChromeTraceJsonlClient(tmp_path)

    client.apply_event(_trace_event())

    lines = client.output_path.read_text().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event == {
        "name": "actor_update",
        "cat": "rl_insight.state",
        "ph": "X",
        "ts": 1500.0,
        "dur": 2250.0,
        "pid": "1234",
        "tid": "rank_2",
        "args": {
            "process_id": "1234",
            "state_lane_id": "rank_2",
            "state_name": "actor_update",
            "monitor.trace_segment": "state_interval",
        },
    }


def test_metrics_are_forwarded_but_not_persisted(tmp_path):
    delegate = Mock()
    client = ChromeTraceJsonlClient(tmp_path, delegate=delegate)
    metric = {"kind": "gauge", "name": "reward", "value": 1.0}

    client.apply_event(metric)

    assert not client.output_path.exists()
    delegate.apply_event.assert_called_once_with(metric)


def test_trace_event_is_also_forwarded(tmp_path):
    delegate = Mock()
    client = ChromeTraceJsonlClient(tmp_path, delegate=delegate)
    event = _trace_event()

    client.apply_event(event)

    assert client.output_path.exists()
    delegate.apply_event.assert_called_once_with(event)


def test_registration_uses_rl_trace_backend_name():
    register = Mock()

    assert register_rl_insight_client(register)

    register.assert_called_once()
    backend_name, factory = register.call_args.args
    assert backend_name == BACKEND_NAME
    assert callable(factory)


def test_factory_uses_configured_output_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path))

    client = create_rl_trace_observer_client(config=object())

    assert client.output_path.parent == tmp_path


def test_default_verl_registration_does_not_install_patch(monkeypatch):
    from rl_trace_observer.integrations.verl.patch import is_installed, uninstall

    if is_installed():
        uninstall()
    monkeypatch.delenv("RL_TRACE_VIZTRACER", raising=False)
    sys.modules.pop("rl_trace_observer.integrations.verl.register", None)

    __import__("rl_trace_observer.integrations.verl.register")

    assert not is_installed()
