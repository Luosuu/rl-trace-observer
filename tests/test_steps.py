import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from rl_trace_observer.integrations.verl import steps
from rl_trace_observer.integrations.verl.import_hook import when_imported
from rl_trace_observer.process_record import record_path


@pytest.fixture
def trainer_module(tmp_path, monkeypatch):
    (tmp_path / "fake_trainer_base.py").write_text(
        "class PPOTrainer:\n"
        "    global_steps = 3\n"
        "    def __init__(self, rollout='vllm', managers=()):\n"
        "        self.config = Config(rollout)\n"
        "        self.managers = list(managers)\n"
        "    def step(self, metrics, timing_raw):\n"
        "        metrics['ran'] = True\n"
        "        return 'batch'\n"
        "    def _rollout_server_managers(self):\n"
        "        return self.managers\n"
        "    def _start_rollout_profiling(self):\n"
        "        for manager in self.managers:\n"
        "            manager.start_profile()\n"
        "class Config:\n"
        "    def __init__(self, rollout):\n"
        "        self.actor_rollout_ref = type('R', (), {'rollout': type('N', (), {'name': rollout})})\n"
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
    when_imported(trainer_module, steps.patch_trainer)
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
    when_imported(trainer_module, steps.patch_trainer)
    module = importlib.import_module(trainer_module)

    with pytest.raises(TypeError):
        module.PPOTrainer().step()
    assert [name for name, _ in spans] == ["global_step"]


def test_already_imported_module_is_patched_immediately(trainer_module, spans):
    module = importlib.import_module(trainer_module)

    when_imported(trainer_module, steps.patch_trainer)

    module.PPOTrainer().step({}, {})
    assert len(spans) == 1


def test_patched_module_keeps_its_source(trainer_module, spans):
    when_imported(trainer_module, steps.patch_trainer)
    module = importlib.import_module(trainer_module)

    # Tracebacks and inspect read the source through the module's loader.
    assert "class PPOTrainer" in module.__loader__.get_source(trainer_module)


class _Manager:
    def __init__(self):
        self.calls = []

    def start_profile(self, **kwargs):
        self.calls.append(kwargs)


@pytest.mark.parametrize(("rollout", "kwargs"), [("tokenspeed", {"global_step": 3}), ("vllm", {})])
def test_tokenspeed_rollout_profiling_is_told_the_step(trainer_module, spans, rollout, kwargs):
    when_imported(trainer_module, steps.patch_trainer)
    module = importlib.import_module(trainer_module)
    manager = _Manager()

    module.PPOTrainer(rollout, [manager])._start_rollout_profiling()

    assert manager.calls == [kwargs]
