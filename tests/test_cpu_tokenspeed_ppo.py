"""Run VERL's real ``main_ppo`` on CPU with ``rollout.name=tokenspeed`` against a fake TokenSpeed server.

Everything but the server is real: VERL builds two hybrid TokenSpeed replicas
through this package's plugin, each server actor launches ``fake_tokenspeed.py``
with ``tokenspeed serve``'s arguments, the training workers sleep, wake and
update the replicas over TokenSpeed's HTTP API (weights over ``gloo``), and
profiled steps register each replica's VizTracer and Proton files.
"""

import json
import sys
from collections import defaultdict

from cpu_ppo.assets import build_assets
from test_cpu_ppo import ROLLOUT_N, STEPS, TESTS_DIR, TRAIN_BATCH_SIZE, WORLD_SIZE, failure_report, rl_insight_spans
from test_cpu_ppo import run_main_ppo as run
from transformers import AutoModelForCausalLM

from rl_trace_observer.merger.cli import main as merge


def _events(log_dir):
    events = defaultdict(list)
    for path in sorted(log_dir.glob("fake-tokenspeed-*.jsonl")):
        for line in path.read_text().splitlines():
            events[path.stem].append(json.loads(line))
    return events


def test_ppo_with_tokenspeed_rollout_on_cpu(tmp_path, load_in_perfetto):
    assets = build_assets(tmp_path / "assets")
    output_dir = tmp_path / "artifacts"
    log_dir = tmp_path / "fake-tokenspeed"
    log = tmp_path / "main_ppo.log"
    env = {
        "RL_TRACE_TOKENSPEED_COMMAND": f"{sys.executable} {TESTS_DIR / 'cpu_ppo' / 'fake_tokenspeed.py'}",
        "FAKE_TOKENSPEED_LOG_DIR": str(log_dir),
    }

    result = run(assets, output_dir, log, rollout="tokenspeed", env=env)
    model = AutoModelForCausalLM.from_pretrained(assets["model"])
    weights = {name: float(tensor.float().sum()) for name, tensor in model.state_dict().items()}
    assert result.returncode == 0, failure_report(log)

    # Every request is one span on a slot lane of its replica.
    generates = [span for span in rl_insight_spans(output_dir) if span["name"] == "tokenspeed_generate"]
    assert len(generates) == TRAIN_BATCH_SIZE * ROLLOUT_N * STEPS
    assert {span["tid"].split("/")[0] for span in generates} == {f"replica_{r}" for r in range(WORLD_SIZE)}

    servers = _events(log_dir)
    assert len(servers) == WORLD_SIZE
    for events in servers.values():
        assert [event for event in events if event["event"] == "error"] == []
        kinds = [event["event"] for event in events]
        # Proton profiles come from one session per scheduler.
        env = events[0]["env"]
        assert env["TOKENSPEED_PROTON_SESSION_DIR"].startswith(str(output_dir / "rollout"))
        assert kinds.count("init_group") == 1
        assert kinds.count("generate") == TRAIN_BATCH_SIZE * ROLLOUT_N * STEPS // WORLD_SIZE
        # One sync after loading the model and one after every step, each
        # sending every weight of the trained model.
        updates = [event for event in events if event["event"] == "update"]
        names = [name for event in updates for name in event["names"]]
        assert len(names) == len(weights) * (STEPS + 1) and set(names) <= set(weights)
        # The test reward is 0 for random tokens, so the weights stay those of the checkpoint.
        expected = sum(weights[name] for name in names)
        assert abs(sum(event["checksum"] for event in updates) - expected) <= 1e-3 * abs(expected)
        # The replica sleeps after each rollout and wakes before the next.
        assert kinds.count("release") >= STEPS
        assert kinds.count("start_profile") == kinds.count("stop_profile") == STEPS

    merged = tmp_path / "step.json"
    assert merge([str(output_dir), "-o", str(merged), "--strict", "--step", str(STEPS)]) == 0
    manifest = json.loads((tmp_path / "step.manifest.json").read_text())
    tokenspeed = [a for a in manifest["artifacts"] if a["kind"].startswith("tokenspeed") and a["selected"]]
    assert sorted((a["role"], a["rank_tag"], a["kind"]) for a in tokenspeed) == [
        (f"rollout_replica{r}", "TP0", kind)
        for r in range(WORLD_SIZE)
        for kind in ("tokenspeed_proton", "tokenspeed_viztracer")
    ]
    [run_id] = manifest["runs"]
    assert all(f"{run_id}-step-{STEPS}" in a["path"] for a in tokenspeed)

    processor = load_in_perfetto(merged)
    lanes = processor.query(
        """
        select p.name as process, t.name as thread from thread t join process p using (upid)
        where p.name like '% · TP0' and t.name is not null
        """
    ).as_pandas_dataframe()
    assert lanes.process.nunique() == WORLD_SIZE
    for _, threads in lanes.groupby("process"):
        assert {thread.split(" · ")[0] for thread in threads.thread} == {"VizTracer", "Proton"}
    flows = processor.query(
        "select count(*) as n from flow f join slice s on f.slice_out = s.id where s.name = 'forward'"
    ).as_pandas_dataframe()
    assert flows.n[0] == WORLD_SIZE

    # Every request of the step is one flow from the agent loop's call to the server that handled it.
    requests = processor.query(
        """
        select extract_arg(o.arg_set_id, 'args.request_id') as caller,
               extract_arg(i.arg_set_id, 'args.request_id') as server,
               po.name as caller_process, pi.name as server_process
        from flow f
        join slice o on f.slice_out = o.id join thread_track tto on o.track_id = tto.id
        join thread tho on tho.utid = tto.utid join process po on po.upid = tho.upid
        join slice i on f.slice_in = i.id join thread_track tti on i.track_id = tti.id
        join thread thi on thi.utid = tti.utid join process pi on pi.upid = thi.upid
        where o.name = 'rollout_request' and i.name = 'tokenspeed_generate'
        """
    ).as_pandas_dataframe()
    generates_in_step = processor.query(
        "select count(*) as n from slice where name = 'tokenspeed_generate'"
    ).as_pandas_dataframe()
    assert len(requests) == generates_in_step.n[0] == TRAIN_BATCH_SIZE * ROLLOUT_N
    assert (requests.caller == requests.server).all() and requests.caller.nunique() == len(requests)
    assert (requests.caller_process != requests.server_process).all()
