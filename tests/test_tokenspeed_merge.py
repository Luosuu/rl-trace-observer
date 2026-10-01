"""Merge the VizTracer and Proton traces of TokenSpeed scheduler ranks."""

import json

import pytest
from trace_artifacts import BASE_NS, BASE_US, TracedProcess, span, step_marker, write_tokenspeed_pair

from rl_trace_observer.merger import merge_sources
from rl_trace_observer.merger.cli import main
from rl_trace_observer.merger.manifest import build_manifest, select
from rl_trace_observer.merger.sources import TOKENSPEED_PROTON, TOKENSPEED_VIZTRACER, classify


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("run-a-step-3-TP0.viztracer.json", TOKENSPEED_VIZTRACER),
        ("run-a-step-3-DP1-CP0-TP3-EXTEND.viztracer.json", TOKENSPEED_VIZTRACER),
        ("run-a-step-3-TP1.proton.chrome_trace", TOKENSPEED_PROTON),
        # Proton's tree formats are not timelines.
        ("run-a-step-3-TP1.proton.hatchet", None),
        ("run-a-step-3.viztracer.json", None),
    ],
)
def test_classify(tmp_path, name, kind):
    assert classify(tmp_path / name) == kind


def _rollout_run(root, ranks=("TP0", "TP1")):
    """A trainer marking step 1 and a rollout server actor that registered its schedulers' profiles."""
    trainer = TracedProcess(root, 10, role="trainer")
    trainer.jsonl([step_marker(1, pid=10, run_id="run-a")])
    server = TracedProcess(root, 30, role="rollout", actor="TokenSpeedServer_0")
    server.jsonl([span("tokenspeed_generate", BASE_US + 5, 50, pid=30, lane="replica_0")])
    for index, rank_tag in enumerate(ranks):
        server.tokenspeed(rank_tag, scheduler_pid=900 + index)
    return trainer, server


def _merge(root, **selection):
    manifest, sources = build_manifest([root])
    return manifest, merge_sources(select(manifest, sources, **selection), manifest.processes)


def _names(trace, kind):
    return {
        event["pid"] if kind == "process_name" else event["tid"]: event["args"]["name"]
        for event in trace["traceEvents"]
        if event["name"] == kind
    }


def test_each_rank_is_one_process_named_after_the_server_actor(tmp_path):
    _, server = _rollout_run(tmp_path)

    manifest, result = _merge(tmp_path, steps=[1])

    assert not manifest.problems
    linked = [artifact for artifact in manifest.artifacts if artifact.kind.startswith("tokenspeed")]
    assert sorted((a.rank_tag, a.kind, a.process, a.global_step) for a in linked) == [
        (rank, kind, server.key, 1) for rank in ("TP0", "TP1") for kind in (TOKENSPEED_PROTON, TOKENSPEED_VIZTRACER)
    ]
    processes = _names(result.trace, "process_name")
    threads = _names(result.trace, "thread_name")
    events = [event for event in result.trace["traceEvents"] if event["ph"] == "X"]
    for rank in ("TP0", "TP1"):
        (pid,) = [
            pid for pid, name in processes.items() if name == f"node-1.example pid 30 · TokenSpeedServer_0 · {rank}"
        ]
        lanes = {threads[event["tid"]] for event in events if event["pid"] == pid}
        assert lanes == {"VizTracer · MainThread", "Proton · CPU Thread 900", "Proton · GPU Stream 7"}


def test_sources_are_placed_by_their_own_anchors(tmp_path):
    _rollout_run(tmp_path, ranks=("TP0",))

    _, result = _merge(tmp_path)

    # Every file was written so that its slices line up on the shared timeline.
    starts = {event["name"]: event["ts"] for event in result.trace["traceEvents"] if event["ph"] == "X"}
    base_us = (result.global_base_ns - BASE_NS) / 1000
    assert starts["forward"] + base_us == 10.0
    assert starts["attention"] + base_us == 12.0
    assert starts["flash_attn_kernel"] + base_us == 20.0


def test_flows_link_a_rank_s_two_files_and_never_cross_ranks(tmp_path):
    _rollout_run(tmp_path)

    _, result = _merge(tmp_path)
    events = result.trace["traceEvents"]
    threads = _names(result.trace, "thread_name")
    pid_of = {event["tid"]: event["pid"] for event in events if event["name"] == "thread_name"}
    flows: dict[int, list[dict]] = {}
    for event in events:
        if event["ph"] in "sf":
            flows.setdefault(event["id"], []).append(event)

    # Per rank: Proton's launch flow and the scope flow for scope 5; the start
    # for scope 6, which Proton did not record, is dropped.
    assert len(flows) == 4
    for ends in flows.values():
        assert sorted(event["ph"] for event in ends) == ["f", "s"]
        start, end = sorted(ends, key=lambda event: event["ph"], reverse=True)
        assert pid_of[start["tid"]] == pid_of[end["tid"]]
        if start["name"] == "viztracer->proton":
            assert (threads[start["tid"]], threads[end["tid"]]) == ("VizTracer · MainThread", "Proton · CPU Thread 900")
            attention = next(e for e in events if e["name"] == "attention" and e["tid"] == end["tid"])
            assert end["ts"] == attention["ts"] and end["bp"] == "e"
        else:
            assert (threads[start["tid"]], threads[end["tid"]]) == ("Proton · CPU Thread 900", "Proton · GPU Stream 7")


def test_unregistered_pair_still_merges_as_one_rank(tmp_path):
    write_tokenspeed_pair(tmp_path, "e0", "TP0")

    manifest, result = _merge(tmp_path)

    assert {problem.kind for problem in manifest.problems} == {"unlinked"}
    assert {a.rank_tag for a in manifest.artifacts} == {"TP0"}
    assert list(_names(result.trace, "process_name").values()) == ["TokenSpeed e0 TP0"]
    assert sum(event["name"] == "viztracer->proton" for event in result.trace["traceEvents"]) == 2


def test_strict_cli_merges_a_profiled_rollout_step(tmp_path):
    (tmp_path / "artifacts").mkdir()
    _rollout_run(tmp_path / "artifacts")
    output = tmp_path / "step1.json"

    assert main([str(tmp_path / "artifacts"), "-o", str(output), "--strict", "--step", "1"]) == 0
    manifest = json.loads(output.with_name("step1.manifest.json").read_text())
    (session,) = manifest["sessions"]
    assert sum("rollout/replica0" in path for path in session["artifacts"]) == 4


def test_matches_tokenspeed_merge_traces(tmp_path):
    """Same timeline and flow links as TokenSpeed's own all-ranks merge."""
    trace_merge = pytest.importorskip("tokenspeed.cli.trace_merge")
    pairs = [
        (
            rank,
            *write_tokenspeed_pair(
                tmp_path, "e0", f"TP{rank}", base_ns=BASE_NS + rank * 7_000, scheduler_pid=900 + rank
            ),
        )
        for rank in (0, 1)
    ]
    official_path = tmp_path / "official.json"
    trace_merge.merge_all_ranks(pairs, official_path)
    official = json.loads(official_path.read_text())

    _, ours = _merge(tmp_path)

    def summary(events, base_ns):
        """Absolute slice times and flow (start, end) times, per flow name."""
        slices = sorted((e["name"], round(base_ns + e["ts"] * 1000)) for e in events if e["ph"] == "X")
        ends: dict = {}
        for event in events:
            if event["ph"] in "sf":
                ends.setdefault(event["id"], {})[event["ph"]] = round(base_ns + event["ts"] * 1000)
        flows = sorted((ids.get("s"), ids.get("f")) for ids in ends.values() if "s" in ids and "f" in ids)
        return slices, flows

    official_base = official["viztracer_metadata"]["baseTimeNanoseconds"]
    assert summary(ours.trace["traceEvents"], ours.global_base_ns) == summary(official["traceEvents"], official_base)


def test_perfetto_binds_flows_to_slices_of_one_rank(tmp_path, load_in_perfetto):
    (tmp_path / "artifacts").mkdir()
    _rollout_run(tmp_path / "artifacts")
    output = tmp_path / "merged.json"
    assert main([str(tmp_path / "artifacts"), "-o", str(output), "--strict"]) == 0

    processor = load_in_perfetto(output)
    assert (
        processor.query("select * from stats where severity in ('error', 'data_loss') and value > 0")
        .as_pandas_dataframe()
        .empty
    )
    flows = processor.query(
        """
        select out.name as source, inn.name as target, t_out.upid = t_in.upid as same_process
        from flow f
        join slice out on f.slice_out = out.id
        join slice inn on f.slice_in = inn.id
        join thread_track tt_out on out.track_id = tt_out.id
        join thread t_out on tt_out.utid = t_out.utid
        join thread_track tt_in on inn.track_id = tt_in.id
        join thread t_in on tt_in.utid = t_in.utid
        """
    ).as_pandas_dataframe()
    assert sorted(zip(flows.source, flows.target, strict=True)) == [
        ("attention", "flash_attn_kernel"),
        ("attention", "flash_attn_kernel"),
        ("forward", "attention"),
        ("forward", "attention"),
    ]
    assert flows.same_process.all()
