"""CPU-only end-to-end tests with VERL's real single controller and plugin loading.

No test here patches VERL or calls the plugin directly: every process loads it
through VERL's ``verl.plugins`` entry point discovery on ``import verl``.
"""

import json
import os

import pytest

ray = pytest.importorskip("ray")
pytest.importorskip("rl_insight.client.base")
pytest.importorskip("verl")

from conftest import load_ray_runtime_env  # noqa: E402
from verl.single_controller.base.decorator import Dispatch, register  # noqa: E402
from verl.single_controller.base.worker import Worker  # noqa: E402
from verl.single_controller.ray.base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup  # noqa: E402
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env  # noqa: E402
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig  # noqa: E402
from verl.utils.tracking import RLInsightLogger  # noqa: E402

WORLD_SIZE = 2


@ray.remote
class _ActorWorker(Worker, DistProfilerExtension):
    def __init__(self):
        Worker.__init__(self)
        profiler = DistProfiler(rank=self.rank, config=ProfilerConfig(enable=True, all_ranks=True))
        DistProfilerExtension.__init__(self, profiler)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @DistProfiler.annotate(role="actor_update")
    def update_actor(self):
        return os.getpid()


@ray.remote
class _RolloutServer:
    """Stands in for VERL's vLLM/SGLang async server, which needs a GPU engine."""

    def __init__(self, replica_rank: int):
        self.replica_rank = replica_rank

    def generate(self):
        with RLInsightLogger.trace_state("vllm_generate", state_lane_id=f"replica_{self.replica_rank}"):
            return os.getpid()


def _start_ray(output_dir, **extra_env):
    # Mirrors verl.trainer.main_ppo: enabling the rl_insight logger sets this in
    # the driver, and get_ppo_ray_runtime_env forwards it to every worker.
    os.environ["VERL_RL_INSIGHT_ENABLE"] = "1"
    runtime_env = load_ray_runtime_env(
        **get_ppo_ray_runtime_env()["env_vars"], RL_TRACE_OUTPUT_DIR=str(output_dir), **extra_env
    )
    ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False, runtime_env=runtime_env)


@pytest.fixture
def verl_env(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", "local://rl-trace-observer")
    monkeypatch.setenv("VERL_RL_INSIGHT_ENABLE", "1")
    yield tmp_path
    RLInsightLogger.finish()
    ray.shutdown()


def _events_by_pid(output_dir):
    events = {}
    for path in output_dir.glob("rl-insight-*.chrome.jsonl"):
        pid = int(path.name.rsplit("-pid-", 1)[1].split(".", 1)[0])
        events[pid] = [json.loads(line) for line in path.read_text().splitlines()]
    return events


def _actor_group():
    return RayWorkerGroup(
        resource_pool=RayResourcePool([WORLD_SIZE], use_gpu=False, max_colocate_count=1),
        ray_cls_with_init=RayClassWithInitArgs(cls=_ActorWorker),
        name_prefix="rl_trace_actor",
    )


def test_driver_actor_and_rollout_processes_write_semantic_spans(verl_env):
    _start_ray(verl_env)

    # Driver: VERL's Tracking creates RLInsightLogger with the trainer config.
    RLInsightLogger(
        "project",
        "experiment",
        config={"trainer": {"rl_insight": {"server": {"backend": "rl_trace_observer"}}}},
    )
    with RLInsightLogger.trace_state("train_batch"):
        actor_pids = _actor_group().update_actor()
        rollout_pid = ray.get(_RolloutServer.remote(0).generate.remote())

    events = _events_by_pid(verl_env)
    assert [event["name"] for event in events[os.getpid()]] == ["train_batch"]
    assert len(set(actor_pids)) == WORLD_SIZE
    assert sorted(events[pid][0]["tid"] for pid in actor_pids) == [f"rank_{rank}" for rank in range(WORLD_SIZE)]
    assert all([event["name"] for event in events[pid]] == ["actor_update"] for pid in actor_pids)
    assert [(event["name"], event["tid"]) for event in events[rollout_pid]] == [("vllm_generate", "replica_0")]


def test_optional_viztracer_follows_verl_profile_window(verl_env):
    pytest.importorskip("viztracer")
    _start_ray(verl_env, RL_TRACE_VIZTRACER="1", RL_TRACE_VIZTRACER_MIN_DURATION_US="0")

    group = _actor_group()
    group.start_profile(role="e2e", profile_step=3)
    group.update_actor()
    # VERL's trainer passes run_command to every stop_profile call.
    group.stop_profile(run_command=False)

    traces = sorted(verl_env.glob("step-3-role-e2e-rank-*.viztracer.json"))
    assert [path.name.split("-rank-")[1].split("-")[0] for path in traces] == [str(rank) for rank in range(WORLD_SIZE)]
