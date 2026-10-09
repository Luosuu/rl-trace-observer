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
        "    def fit(self, agent_loop_manager):\n"
        "        return 'trained'\n"
        "    def _rollout_server_managers(self):\n"
        "        return self.managers\n"
        "    def _start_rollout_profiling(self):\n"
        "        for manager in self.managers:\n"
        "            manager.start_profile()\n"
        "class Config:\n"
        "    def __init__(self, rollout):\n"
        "        self.actor_rollout_ref = type('R', (), {'rollout': type('N', (), {'name': rollout})})\n"
        "        self.global_profiler = {'steps': [3]}\n"
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


def test_fit_waits_for_profiles_written_in_the_background(trainer_module, spans, monkeypatch):
    waited = []
    monkeypatch.setattr(steps, "_wait_for_profiles", waited.append)
    when_imported(trainer_module, steps.patch_trainer)
    module = importlib.import_module(trainer_module)
    trainer = module.PPOTrainer()

    assert trainer.fit(None) == "trained"
    assert waited == [trainer]


def test_a_missing_verl_method_fails_clearly():
    module = types.ModuleType("fake_trainer_base_old")
    module.PPOTrainer = type("PPOTrainer", (), {"step": lambda self: None, "fit": lambda self: None})

    with pytest.raises(RuntimeError, match="supports verl 0.9.1.*_start_rollout_profiling"):
        steps.patch_trainer(module)


def test_one_step_off_starts_a_profile_only_after_the_previous_stop(spans, monkeypatch):
    import asyncio

    import ray

    gets = []
    monkeypatch.setattr(ray, "get", lambda refs: gets.append(list(refs)))

    class Remote:
        def __init__(self, name):
            self.name = name

        def remote(self, **kwargs):
            return (self.name, kwargs.get("global_step"))

    server = types.SimpleNamespace(start_profile=Remote("start"), stop_profile=Remote("stop"))

    class Trainer:
        global_steps = 2

        def __init__(self):
            self.config = types.SimpleNamespace(
                actor_rollout_ref=types.SimpleNamespace(rollout=types.SimpleNamespace(name="tokenspeed")),
                global_profiler={"steps": [2, 3]},
            )
            self.llm_server_manager = types.SimpleNamespace(
                get_replicas=lambda: [types.SimpleNamespace(servers=[server])]
            )

        async def fit(self):
            pass

        async def fit_step(self):
            task = asyncio.create_task(self._async_gen_next_batch())
            await asyncio.sleep(0)
            await task
            self.global_steps += 1

        async def _async_gen_next_batch(self):
            return "batch"

    module = types.ModuleType("fake_one_step_off")
    module.OneStepOffRayTrainer = Trainer
    assert steps.patch_one_step_off_trainer(module)
    trainer = Trainer()

    async def two_steps():
        await trainer.fit_step()
        await trainer.fit_step()

    asyncio.run(two_steps())

    # Step 3's start waits for step 2's stop, which was not awaited.
    assert gets == [[], [("start", 2)], [("stop", None)], [("start", 3)]]
