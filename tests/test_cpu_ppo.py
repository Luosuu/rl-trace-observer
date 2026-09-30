"""Run VERL's real ``main_ppo`` for one step on CPU and merge its traces.

``cpu_ppo.cpu_plugin`` supplies a cpu platform and a mock rollout through
VERL's extension points; ``cpu_ppo.assets`` builds a tiny model, tokenizer and
dataset offline. Every VERL process loads both that plugin and this package's
own plugin, exactly as in a real run.
"""

import json
import os
import socket
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from cpu_ppo.assets import build_assets

from rl_trace_observer.merger.cli import main as merge
from rl_trace_observer.output import safe_component

TESTS_DIR = Path(__file__).resolve().parent
WORLD_SIZE = 2
TOLERANCE_NS = 5_000_000


def _run_main_ppo(assets: dict[str, Path], output_dir: Path, log: Path) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(TESTS_DIR), os.environ.get("PYTHONPATH")])),
        "VERL_PLATFORM": "cpu",
        "VERL_USE_EXTERNAL_MODULES": "cpu_ppo.cpu_plugin",
        "RL_TRACE_OUTPUT_DIR": str(output_dir),
        "RL_INSIGHT_SERVER_URL": "local://rl-trace-observer",
        "TOKENIZERS_PARALLELISM": "false",
        "HYDRA_FULL_ERROR": "1",
    }
    overrides = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={assets['train']}",
        f"data.val_files={assets['val']}",
        "data.train_batch_size=4",
        "data.max_prompt_length=32",
        "data.max_response_length=16",
        "data.dataloader_num_workers=0",
        f"actor_rollout_ref.model.path={assets['model']}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        # FSDP1 treats the integer device id as a CUDA index; FSDP2 follows the device mesh.
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.ref.strategy=fsdp2",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.rollout.name=mock",
        "actor_rollout_ref.rollout.n=2",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        # RL-Insight rollout metrics require log stats.
        "actor_rollout_ref.rollout.disable_log_stats=False",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "reward.num_workers=1",
        f"reward.custom_reward_function.path={TESTS_DIR / 'cpu_ppo' / 'reward.py'}",
        "global_profiler.tool=torch",
        "global_profiler.steps=[1]",
        f"global_profiler.save_path={output_dir / 'torch'}",
        "actor_rollout_ref.actor.profiler.enable=True",
        "actor_rollout_ref.actor.profiler.all_ranks=True",
        "trainer.device=cpu",
        f"trainer.n_gpus_per_node={WORLD_SIZE}",
        "trainer.nnodes=1",
        "trainer.total_training_steps=1",
        'trainer.logger=["console","rl_insight"]',
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.project_name=rl_trace_observer",
        "trainer.experiment_name=cpu_ppo",
        # TransferQueue storage units and the placement groups each hold a
        # logical CPU or GPU; declare enough of both for a small machine.
        "transfer_queue.backend.SimpleStorage.num_data_storage_units=2",
        "+ray_kwargs.ray_init.num_cpus=16",
        f"+ray_kwargs.ray_init.num_gpus={WORLD_SIZE}",
        # Keep Hydra's run directory out of the checkout.
        f"hydra.run.dir={log.parent / 'hydra'}",
    ]
    with log.open("w") as output:
        return subprocess.run(
            [sys.executable, "-m", "verl.trainer.main_ppo", *overrides],
            env=env,
            cwd=log.parent,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
        )


def _spans_by_lane(output_dir: Path) -> dict[str, set[str]]:
    lanes = defaultdict(set)
    for path in output_dir.glob("rl-insight-*.chrome.jsonl"):
        for event in map(json.loads, path.read_text().splitlines()):
            lanes[event["tid"]].add(event["name"])
    return lanes


def test_one_ppo_step_on_cpu_produces_aligned_traces(tmp_path, load_in_perfetto):
    assets = build_assets(tmp_path / "assets")
    output_dir = tmp_path / "artifacts"
    log = tmp_path / "main_ppo.log"

    result = _run_main_ppo(assets, output_dir, log)
    assert result.returncode == 0, log.read_text()[-5000:]

    # Every actor rank and rollout replica wrote its own semantic lane.
    lanes = _spans_by_lane(output_dir)
    for rank in range(WORLD_SIZE):
        assert {"actor_compute_log_prob", "actor_update"} <= lanes[f"rank_{rank}"]
        assert lanes[f"replica_{rank}"] == {"mock_generate"}
    assert len(list((output_dir / "torch").glob("*.json.gz"))) == WORLD_SIZE

    merged = tmp_path / "merged.json"
    assert merge([str(output_dir), "-o", str(merged), "--strict"]) == 0

    processor = load_in_perfetto(merged)
    slices = processor.query(
        """
        select s.ts, s.dur, p.name as process
        from slice s
        join thread_track tt on s.track_id = tt.id
        join thread using (utid)
        join process p using (upid)
        where s.name = 'actor_update' and s.depth = 0
        """
    ).as_pandas_dataframe()
    assert len(slices) == 2 * WORLD_SIZE
    for rank in range(WORLD_SIZE):
        pid = next(path.name.split("pid")[1].split("_")[0] for path in (output_dir / "torch").glob(f"*rank{rank}-*"))
        rl_insight = slices[slices.process == f"RL-Insight {safe_component(socket.gethostname())} pid {pid}"].iloc[0]
        torch_span = slices[slices.process.str.startswith("Torch") & slices.process.str.contains(f"pid {pid}")].iloc[0]
        assert abs(torch_span.ts - rl_insight.ts) < TOLERANCE_NS
        assert abs((torch_span.ts + torch_span.dur) - (rl_insight.ts + rl_insight.dur)) < TOLERANCE_NS
