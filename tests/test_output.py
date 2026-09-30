import pytest

from rl_trace_observer.integrations.rl_insight import create_rl_trace_observer_client
from rl_trace_observer.output import safe_component, trace_output_dir


def test_output_dir_is_created(tmp_path, monkeypatch):
    output_dir = tmp_path / "nested" / "artifacts"
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(output_dir))

    assert trace_output_dir() == output_dir
    assert output_dir.is_dir()


def test_unset_output_dir_is_rejected(monkeypatch):
    monkeypatch.delenv("RL_TRACE_OUTPUT_DIR", raising=False)

    with pytest.raises(ValueError, match="must be set"):
        trace_output_dir()


@pytest.mark.parametrize("value", ["rl_trace_outputs", "./artifacts", "  "])
def test_relative_or_blank_output_dir_is_rejected(value, monkeypatch):
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", value)

    with pytest.raises(ValueError):
        trace_output_dir()


def test_rl_insight_client_requires_output_dir(monkeypatch):
    # Fails when RL-Insight creates the client, i.e. on the first span of a
    # misconfigured worker, instead of writing into Ray's temporary working_dir.
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", "rl_trace_outputs")

    with pytest.raises(ValueError, match="absolute"):
        create_rl_trace_observer_client(config=object())


def test_safe_component_replaces_separators():
    assert safe_component("node-1.example/x y") == "node-1.example_x_y"
    assert safe_component("///") == "unknown"
