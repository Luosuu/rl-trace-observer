"""Run VERL's real ``main_ppo`` for one step on CPU and merge its traces.

``cpu_ppo.cpu_plugin`` supplies a cpu platform and a mock rollout through
VERL's extension points; ``cpu_ppo.assets`` builds a tiny model, tokenizer and
dataset offline. Every VERL process loads both that plugin and this package's
own plugin, exactly as in a real run.
"""

import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from cpu_ppo.assets import build_assets

from rl_trace_observer.merger.cli import main as merge

TESTS_DIR = Path(__file__).resolve().parent
WORLD_SIZE = 2
TRAIN_BATCH_SIZE = 4
STEPS = 2
ROLLOUT_N = 2
# MockLLMServer.PREFILL_SECONDS: every mock request lasts at least this long.
MIN_GENERATE_US = 50_000
TOLERANCE_NS = 5_000_000


def run_main_ppo(
    assets: dict[str, Path],
    output_dir: Path,
    log: Path,
    rollout: str = "mock",
    env: dict[str, str] | None = None,
    entry: str = "verl.trainer.main_ppo",
    overrides: dict[str, object] | None = None,
) -> subprocess.CompletedProcess:
    """Run a VERL entry point on CPU; ``overrides`` replaces or adds Hydra overrides by key."""
    extra_env = env or {}
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(TESTS_DIR), os.environ.get("PYTHONPATH")])),
        "VERL_PLATFORM": "cpu",
        "VERL_USE_EXTERNAL_MODULES": "cpu_ppo.cpu_plugin",
        "RL_TRACE_OUTPUT_DIR": str(output_dir),
        "RL_INSIGHT_SERVER_URL": "local://rl-trace-observer",
        "TOKENIZERS_PARALLELISM": "false",
        "HYDRA_FULL_ERROR": "1",
        **extra_env,
    }
    arguments = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={assets['train']}",
        f"data.val_files={assets['val']}",
        f"data.train_batch_size={TRAIN_BATCH_SIZE}",
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
        f"actor_rollout_ref.rollout.name={rollout}",
        f"actor_rollout_ref.rollout.n={ROLLOUT_N}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
        # RL-Insight rollout metrics require log stats.
        "actor_rollout_ref.rollout.disable_log_stats=False",
        "actor_rollout_ref.rollout.agent.num_workers=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4",
        "reward.num_workers=1",
        f"reward.custom_reward_function.path={TESTS_DIR / 'cpu_ppo' / 'reward.py'}",
        "global_profiler.tool=torch",
        f"global_profiler.steps={list(range(1, STEPS + 1))}",
        f"global_profiler.save_path={output_dir / 'torch'}",
        "actor_rollout_ref.actor.profiler.enable=True",
        "actor_rollout_ref.actor.profiler.all_ranks=True",
        "trainer.device=cpu",
        f"trainer.n_gpus_per_node={WORLD_SIZE}",
        "trainer.nnodes=1",
        f"trainer.total_training_steps={STEPS}",
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
    replaced = overrides or {}
    keys = {key.lstrip("+") for key in replaced}
    arguments = [o for o in arguments if o.split("=", 1)[0].lstrip("+") not in keys]
    arguments += [f"{key}={value}" for key, value in replaced.items()]
    with log.open("w") as output:
        return subprocess.run(
            [sys.executable, "-m", entry, *arguments],
            env=env,
            cwd=log.parent,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=900,
            check=False,
        )


def failure_report(log: Path) -> str:
    # Ray interleaves worker output, and a C++ stack trace easily pushes the
    # actual error out of the tail; list the error lines first.
    lines = log.read_text(errors="replace").splitlines()
    markers = ("Error", "Exception", "enforce fail", "Traceback", "Killed", "OOM")
    errors = list(dict.fromkeys(line for line in lines if any(marker in line for marker in markers)))
    return "\n".join(["--- error lines ---", *errors[:80], "--- log tail ---", *lines[-60:]])


def rl_insight_spans(output_dir: Path) -> list[dict]:
    return [
        event
        for path in output_dir.glob("rl-insight-*.chrome.jsonl")
        for event in map(json.loads, path.read_text().splitlines())
    ]


def test_one_ppo_step_on_cpu_produces_aligned_traces(tmp_path, load_in_perfetto):
    assets = build_assets(tmp_path / "assets")
    output_dir = tmp_path / "artifacts"
    log = tmp_path / "main_ppo.log"

    result = run_main_ppo(assets, output_dir, log)
    assert result.returncode == 0, failure_report(log)

    spans = rl_insight_spans(output_dir)
    lanes = defaultdict(set)
    for span in spans:
        lanes[span["tid"]].add(span["name"])
    # Every actor rank wrote its own semantic lane.
    for rank in range(WORLD_SIZE):
        assert {"actor_compute_log_prob", "actor_update"} <= lanes[f"rank_{rank}"]
    # Every rollout request is one span on a batch-slot lane of its replica.
    generates = [span for span in spans if span["name"] == "mock_generate"]
    assert len(generates) == TRAIN_BATCH_SIZE * ROLLOUT_N * STEPS
    assert len({span["args"]["request_id"] for span in generates}) == len(generates)
    assert {span["tid"].split("/")[0] for span in generates} == {f"replica_{rank}" for rank in range(WORLD_SIZE)}
    assert all(span["dur"] >= MIN_GENERATE_US for span in generates)
    assert len(list((output_dir / "torch").glob("*.json.gz"))) == WORLD_SIZE * STEPS
    # The trainer marks every step.
    step_spans = sorted(span["args"]["global_step"] for span in spans if span["name"] == "global_step")
    assert step_spans == list(range(1, STEPS + 1))

    merged = tmp_path / "merged.json"
    assert merge([str(output_dir), "-o", str(merged), "--strict"]) == 0

    # Every tracing process registered itself; the Torch traces link to the actor processes.
    manifest = json.loads((tmp_path / "merged.manifest.json").read_text())
    assert manifest["problems"] == []
    # All processes share one run id, and every profiled step is a session with a
    # time window and one Torch trace per rank.
    [run_id] = manifest["runs"]
    assert run_id.startswith("session_") and "-job-" in run_id
    assert {process["run_id"] for process in manifest["processes"]} == {run_id}
    assert [process["role"] for process in manifest["processes"]].count("trainer") == 1
    assert [session["global_step"] for session in manifest["sessions"]] == list(range(1, STEPS + 1))
    for session in manifest["sessions"]:
        assert session["id"] == f"{run_id}-step-{session['global_step']}"
        assert session["start_time_ns"] < session["end_time_ns"]
        assert len(session["artifacts"]) == WORLD_SIZE
    processes = {process["key"]: process for process in manifest["processes"]}
    names = {process["actor_name"] for process in processes.values()}
    assert {f"mock_server_{replica}" for replica in range(WORLD_SIZE)} <= names
    actor_ranks = {process["rank"]: key for key, process in processes.items() if process["rank"] is not None}
    assert set(actor_ranks) == set(range(WORLD_SIZE))
    torch_artifacts = [artifact for artifact in manifest["artifacts"] if artifact["kind"] == "torch"]
    assert {artifact["process"] for artifact in torch_artifacts} == set(actor_ranks.values())
    assert all("torch" in processes[artifact["process"]]["versions"] for artifact in torch_artifacts)

    # One step on its own: its Torch traces and the RL-Insight spans in its window.
    step = STEPS
    merged = tmp_path / "step.json"
    assert merge([str(output_dir), "-o", str(merged), "--strict", "--step", str(step)]) == 0
    selected = json.loads((tmp_path / "step.manifest.json").read_text())
    assert {artifact["global_step"] for artifact in selected["artifacts"] if artifact["selected"]} <= {None, step}
    step_events = [event for event in json.loads(merged.read_text())["traceEvents"] if event["ph"] == "X"]
    assert sum(event["name"] == "mock_generate" for event in step_events) == TRAIN_BATCH_SIZE * ROLLOUT_N
    assert [event["args"]["global_step"] for event in step_events if event["name"] == "global_step"] == [step]

    processor = load_in_perfetto(merged)
    empty_processes = processor.query(
        """
        select p.name from process p
        where p.pid > 0 and not exists (
            select 1 from thread t join thread_track tt using (utid) join slice s on s.track_id = tt.id
            where t.upid = p.upid
        )
        """
    ).as_pandas_dataframe()
    assert empty_processes.empty, empty_processes.name.tolist()
    slices = processor.query(
        """
        select s.ts, s.dur, t.name as thread, t.upid, p.name as process
        from slice s
        join thread_track tt on s.track_id = tt.id
        join thread t using (utid)
        join process p using (upid)
        where s.name = 'actor_update' and s.depth = 0
        """
    ).as_pandas_dataframe()
    assert len(slices) == 2 * WORLD_SIZE
    for rank in range(WORLD_SIZE):
        process = processes[actor_ranks[rank]]
        own = slices[slices.process.str.startswith(f"{process['hostname']} pid {process['os_pid']} ")]
        rl_insight = own[own.thread.str.startswith("RL-Insight")].iloc[0]
        torch_span = own[own.thread.str.startswith("Torch")].iloc[0]
        # Both profilers' view of the step sit in the same Perfetto process and line up.
        assert rl_insight.upid == torch_span.upid
        assert abs(torch_span.ts - rl_insight.ts) < TOLERANCE_NS
        assert abs((torch_span.ts + torch_span.dur) - (rl_insight.ts + rl_insight.dur)) < TOLERANCE_NS
