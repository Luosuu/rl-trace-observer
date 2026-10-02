"""VERL's one-step-off-policy trainer on CPU with standalone TokenSpeed replicas (a fake server).

Training and rollout run on separate (logical) GPUs: two training workers and
two standalone TokenSpeed replicas with TP=1. Step ``n`` trains on the batch
generated during step ``n - 1`` while the replicas generate the batch of step
``n + 1``. Weights reach both replicas through the ``tokenspeed`` checkpoint
engine: training rank 0 broadcasts them once over one ``gloo`` group with both
servers. Only step 2 is profiled; its trace holds the actor update and the
generation that ran beside it.
"""

import json
import sys

from cpu_ppo.assets import build_assets
from test_cpu_ppo import ROLLOUT_N, TESTS_DIR, TRAIN_BATCH_SIZE, failure_report, rl_insight_spans
from test_cpu_ppo import run_main_ppo as run
from test_cpu_tokenspeed_ppo import _events
from transformers import AutoModelForCausalLM

from rl_trace_observer.merger.cli import main as merge

STEPS = 3
PROFILED_STEP = 2
TRAINER_GPUS = ROLLOUT_GPUS = 2


def test_one_step_off_policy_with_standalone_tokenspeed_on_cpu(tmp_path):
    assets = build_assets(tmp_path / "assets")
    output_dir = tmp_path / "artifacts"
    log_dir = tmp_path / "fake-tokenspeed"
    log = tmp_path / "main_ppo.log"
    env = {
        "RL_TRACE_TOKENSPEED_COMMAND": f"{sys.executable} {TESTS_DIR / 'cpu_ppo' / 'fake_tokenspeed.py'}",
        "FAKE_TOKENSPEED_LOG_DIR": str(log_dir),
    }
    overrides = {
        # The one-step-off config extends ppo_trainer through a path relative to a VERL checkout.
        "hydra.searchpath": "[pkg://verl.trainer.config]",
        "actor_rollout_ref.hybrid_engine": False,
        "actor_rollout_ref.rollout.checkpoint_engine.backend": "tokenspeed",
        "trainer.n_gpus_per_node": TRAINER_GPUS,
        "rollout.nnodes": 1,
        "rollout.n_gpus_per_node": ROLLOUT_GPUS,
        "trainer.total_training_steps": STEPS,
        "global_profiler.steps": f"[{PROFILED_STEP}]",
        "+ray_kwargs.ray_init.num_gpus": TRAINER_GPUS + ROLLOUT_GPUS,
        # Its task runner alone reserves 10 CPUs.
        "+ray_kwargs.ray_init.num_cpus": 32,
    }
    result = run(
        assets,
        output_dir,
        log,
        rollout="tokenspeed",
        env=env,
        entry="verl.experimental.one_step_off_policy.main_ppo",
        overrides=overrides,
    )
    assert result.returncode == 0, failure_report(log)
    model = AutoModelForCausalLM.from_pretrained(assets["model"])
    weights = {name: float(tensor.float().sum()) for name, tensor in model.state_dict().items()}

    servers = _events(log_dir)
    assert len(servers) == ROLLOUT_GPUS
    for events in servers.values():
        assert [event for event in events if event["event"] == "error"] == []
        kinds = [event["event"] for event in events]
        # One group with the trainer and both servers, kept across updates.
        [group] = [event for event in events if event["event"] == "init_group"]
        assert group["world_size"] == 1 + ROLLOUT_GPUS
        # Standalone replicas keep their memory.
        assert "release" not in kinds
        # One sync after loading the model and one in every step.
        updates = [event for event in events if event["event"] == "update"]
        names = [name for event in updates for name in event["names"]]
        assert len(names) == len(weights) * (STEPS + 1) and set(names) <= set(weights)
        expected = sum(weights[name] for name in names)
        assert abs(sum(event["checksum"] for event in updates) - expected) <= 1e-3 * abs(expected)
        assert kinds.count("start_profile") == kinds.count("stop_profile") == 1
    offsets = [event["rank_offset"] for events in servers.values() for event in events if "rank_offset" in event]
    assert sorted(offsets) == [1, 2]

    # Step 2 generates the batch of step 3 beside its own actor update.
    spans = rl_insight_spans(output_dir)
    [window] = [s for s in spans if s["name"] == "global_step" and s["args"].get("global_step") == PROFILED_STEP]
    start, end = window["ts"], window["ts"] + window["dur"]
    generates = [s for s in spans if s["name"] == "tokenspeed_generate" and start <= s["ts"] < end]
    assert len(generates) == TRAIN_BATCH_SIZE * ROLLOUT_N
    updates = [s for s in spans if s["name"] == "actor_update" and start <= s["ts"] < end]
    assert updates
    first_generate, last_generate = min(s["ts"] for s in generates), max(s["ts"] + s["dur"] for s in generates)
    assert any(u["ts"] < last_generate and first_generate < u["ts"] + u["dur"] for u in updates), (
        "the rollout of step 3 should overlap the actor update of step 2"
    )

    merged = tmp_path / "step.json"
    assert merge([str(output_dir), "-o", str(merged), "--strict", "--step", str(PROFILED_STEP)]) == 0
    manifest = json.loads((tmp_path / "step.manifest.json").read_text())
    tokenspeed = [a for a in manifest["artifacts"] if a["kind"].startswith("tokenspeed") and a["selected"]]
    kinds = ("tokenspeed_proton", "tokenspeed_viztracer")
    assert sorted((a["role"], a["kind"]) for a in tokenspeed) == [
        (f"rollout_replica{r}", kind) for r in range(ROLLOUT_GPUS) for kind in kinds
    ]
