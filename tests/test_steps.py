import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from rl_trace_observer.integrations.verl import steps
from rl_trace_observer.process_record import record_path


@pytest.fixture
def trainer_module(tmp_path, monkeypatch):
    (tmp_path / "fake_trainer_base.py").write_text(
        "class PPOTrainer:\n"
        "    global_steps = 3\n"
        "    def step(self, metrics, timing_raw):\n"
        "        metrics['ran'] = True\n"
        "        return 'batch'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "fake_trainer_base"
    sys.modules.pop("fake_trainer_base", None)
    sys.meta_path[:] = [finder for finder in sys.meta_path if getattr(finder, "name", None) != "fake_trainer_base"]


@pytest.fixture
def spans(monkeypatch, tmp_path):
    recorded = []

    class Logger:
        @staticmethod
        def enabled():
            return True

        @staticmethod
        def trace_span(name, **kwargs):
            recorded.append((name, kwargs))

    monkeypatch.setitem(sys.modules, "verl.utils.tracking", types.SimpleNamespace(RLInsightLogger=Logger))
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("RL_TRACE_RUN_ID", "run-a")
    return recorded


def test_step_is_marked_once_the_trainer_module_is_imported(trainer_module, spans):
    steps.when_imported(trainer_module, steps.patch_trainer)
    module = importlib.import_module(trainer_module)

    metrics = {}
    assert module.PPOTrainer().step(metrics, {}) == "batch"
    assert metrics == {"ran": True}
    [(name, span)] = spans
    assert name == "global_step"
    assert span["attributes"] == {
        "state_lane_id": "trainer",
        "global_step": 3,
        "run_id": "run-a",
        "rl_trace_observer.step_marker": True,
    }
    assert span["start_time_ns"] <= span["end_time_ns"]
    # Patching twice does not wrap twice.
    assert steps.patch_trainer(module) is False
    # The process that marks steps is the trainer.
    record = json.loads(record_path(Path(os.environ["RL_TRACE_OUTPUT_DIR"])).read_text())
    assert record["role"] == "trainer" and record["run_id"] == "run-a"


def test_step_is_marked_even_when_it_fails(trainer_module, spans):
    steps.when_imported(trainer_module, steps.patch_trainer)
    module = importlib.import_module(trainer_module)

    with pytest.raises(TypeError):
        module.PPOTrainer().step()
    assert [name for name, _ in spans] == ["global_step"]


def test_already_imported_module_is_patched_immediately(trainer_module, spans):
    module = importlib.import_module(trainer_module)

    steps.when_imported(trainer_module, steps.patch_trainer)

    module.PPOTrainer().step({}, {})
    assert len(spans) == 1


def test_patched_module_keeps_its_source(trainer_module, spans):
    steps.when_imported(trainer_module, steps.patch_trainer)
    module = importlib.import_module(trainer_module)

    # Tracebacks and inspect read the source through the module's loader.
    assert "class PPOTrainer" in module.__loader__.get_source(trainer_module)
