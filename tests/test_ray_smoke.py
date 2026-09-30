"""Multi-process smoke test: every Ray worker must write its own semantic artifact."""

import json

import pytest

ray = pytest.importorskip("ray")
pytest.importorskip("rl_insight.client.base")


@ray.remote
class _Worker:
    def run(self, lane_id: str) -> int:
        import os

        # Mirrors VERL_USE_EXTERNAL_MODULES loading the plugin in each worker.
        import rl_insight

        import rl_trace_observer.integrations.verl.register  # noqa: F401

        # VERL workers lazily initialize RL-Insight without the trainer config.
        rl_insight.init()
        with rl_insight.trace_state("actor_update", state_lane_id=lane_id):
            pass
        return os.getpid()


@pytest.fixture
def ray_cluster(tmp_path):
    ray.init(
        num_cpus=2,
        include_dashboard=False,
        log_to_driver=False,
        runtime_env={
            "env_vars": {
                "RL_TRACE_OUTPUT_DIR": str(tmp_path),
                "RL_INSIGHT_SERVER_URL": "local://rl-trace-observer",
            }
        },
    )
    yield tmp_path
    ray.shutdown()


def test_each_ray_worker_writes_semantic_artifact(ray_cluster):
    workers = [_Worker.remote() for _ in range(2)]
    pids = ray.get([worker.run.remote(f"rank_{rank}") for rank, worker in enumerate(workers)])

    artifacts = sorted(ray_cluster.glob("*.chrome.jsonl"))
    assert len(set(pids)) == 2
    assert len(artifacts) == 2
    for artifact in artifacts:
        events = [json.loads(line) for line in artifact.read_text().splitlines()]
        assert [event["name"] for event in events] == ["actor_update"]
    assert {path.name.rsplit("-pid-", 1)[1].split(".")[0] for path in artifacts} == {str(pid) for pid in pids}
