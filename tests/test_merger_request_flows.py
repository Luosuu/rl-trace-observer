"""Request flows: one flow per rollout request across the processes on its path."""

import json

import pytest

from rl_trace_observer.context import REQUEST_FLOW_ATTRIBUTE
from rl_trace_observer.merger import discover, merge_sources, read_source

BASE_US = 1_790_000_000_000_000


def _span(name, ts_us, pid, lane, request_id=None, marked=True, dur=10.0):
    args = {"request_id": request_id} if request_id else {}
    if marked and request_id:
        args[REQUEST_FLOW_ATTRIBUTE] = True
    return {"name": name, "ph": "X", "ts": BASE_US + ts_us, "dur": dur, "pid": pid, "tid": lane, "args": args}


def _write(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def _merge(tmp_path):
    _write(
        tmp_path / "rl-insight-node-1-pid-10.chrome.jsonl",
        [
            _span("rollout_request", 0, "10", "agent_loop/slot_0", "r1", dur=100),
            _span("rollout_request", 5, "10", "agent_loop/slot_1", "r2", dur=100),
            # Same request id, but not marked as a point on a request's path.
            _span("rollout_request", 50, "10", "agent_loop/slot_2", "r3", marked=False),
        ],
    )
    _write(
        tmp_path / "rl-insight-node-1-pid-20.chrome.jsonl",
        [
            _span("tokenspeed_generate", 10, "20", "replica_0/slot_0", "r1", dur=80),
            _span("tokenspeed_generate", 12, "20", "replica_0/slot_1", "r2", dur=80),
            _span("tokenspeed_generate", 60, "20", "replica_0/slot_2", "r3"),
            _span("tokenspeed_generate", 70, "20", "replica_0/slot_3", "alone"),
        ],
    )
    return merge_sources(read_source(kind, path) for kind, path in discover([tmp_path])).trace


def test_each_request_is_one_flow_from_the_agent_loop_to_the_server(tmp_path):
    events = _merge(tmp_path)["traceEvents"]
    # Each flow event binds to the span that starts at its timestamp on its thread.
    spans = {(e["tid"], e["ts"]): e for e in events if e["ph"] == "X"}
    flows = {}
    for event in events:
        if event.get("cat") == "rl_trace_observer.request":
            flows.setdefault(event["id"], []).append(event)

    paths = {}
    for flow in flows.values():
        start, end = sorted(flow, key=lambda event: event["ts"])
        caller, server = spans[(start["tid"], start["ts"])], spans[(end["tid"], end["ts"])]
        assert (start["ph"], end["ph"], end["bp"]) == ("s", "f", "e")
        assert caller["args"]["request_id"] == server["args"]["request_id"]
        assert caller["pid"] != server["pid"]
        paths[caller["args"]["request_id"]] = (caller["name"], server["name"])
    # r3 is not marked on the agent loop's side; "alone" has a single point.
    assert paths == {request: ("rollout_request", "tokenspeed_generate") for request in ("r1", "r2")}


def test_request_flows_connect_slices_in_perfetto(tmp_path, load_in_perfetto):
    merged = tmp_path / "merged.json"
    merged.write_text(json.dumps(_merge(tmp_path)))

    flows = (
        load_in_perfetto(merged)
        .query(
            """
            select o.name as out_name, i.name as in_name,
                   extract_arg(o.arg_set_id, 'args.request_id') as out_request,
                   extract_arg(i.arg_set_id, 'args.request_id') as in_request
            from flow f join slice o on f.slice_out = o.id join slice i on f.slice_in = i.id
            """
        )
        .as_pandas_dataframe()
    )
    assert sorted(zip(flows.out_request, flows.in_request, strict=True)) == [("r1", "r1"), ("r2", "r2")]
    assert set(flows.out_name) == {"rollout_request"} and set(flows.in_name) == {"tokenspeed_generate"}


def _rollout_with_forwards(root):
    """An agent loop call, the server's handling of it, and the forwards two TP ranks ran for it."""
    from trace_artifacts import BASE_US, TracedProcess, span, step_marker

    TracedProcess(root, 10, role="trainer").jsonl([step_marker(1, pid=10, run_id="run-a")])
    point = {"request_id": "r1", REQUEST_FLOW_ATTRIBUTE: True}
    agent = TracedProcess(root, 20, actor="agent_loop_worker_0")
    agent.jsonl([span("rollout_request", BASE_US + 1, 200, pid=20, lane="agent_loop/slot_0", **point)])
    server = TracedProcess(root, 30, role="rollout_server", actor="tokenspeed_server_rollout_0")
    server.jsonl([span("tokenspeed_generate", BASE_US + 2, 190, pid=30, lane="replica_0/slot_0", **point)])
    forwards = [(10.0, 5.0, ["r1", "r2"]), (40.0, 5.0, ["r1", "r2"]), (70.0, 5.0, ["r1"]), (100.0, 5.0, ["r2"])]
    for index, rank_tag in enumerate(("TP0", "TP1")):
        server.tokenspeed(rank_tag, scheduler_pid=900 + index, forwards=forwards)


def _request_path(trace):
    """(process, slice) of each point of the request's flow, in order."""
    events = trace["traceEvents"]
    processes = {e["pid"]: e["args"]["name"] for e in events if e.get("name") == "process_name"}
    slices = {(e["tid"], e["ts"]): e["name"] for e in events if e["ph"] == "X"}
    flows = {}
    for event in events:
        if event.get("cat") == "rl_trace_observer.request":
            flows.setdefault(event["id"], []).append(event)
    paths = [sorted(flow, key=lambda e: e["ts"]) for flow in flows.values()]
    [path] = [p for p in paths if slices[(p[0]["tid"], p[0]["ts"])] == "rollout_request"] or [[]]
    assert [e["ph"] for e in path] == ["s", *["t"] * (len(path) - 2), "f"]
    return [(processes[e["pid"]].rsplit(" · ", 1)[-1], slices[(e["tid"], e["ts"])], e["ts"]) for e in path]


@pytest.mark.parametrize(("every_forward", "forwards"), [(False, [10.0, 70.0]), (True, [10.0, 40.0, 70.0])])
def test_a_request_continues_through_the_forwards_of_one_rank(tmp_path, every_forward, forwards):
    from trace_artifacts import BASE_NS

    from rl_trace_observer.merger.manifest import build_manifest

    _rollout_with_forwards(tmp_path)
    manifest, sources = build_manifest([tmp_path])
    result = merge_sources(sources, manifest.processes, every_forward=every_forward)

    path = _request_path(result.trace)
    base_us = (BASE_NS - result.global_base_ns) / 1000
    assert [step[:2] for step in path[:2]] == [
        ("agent_loop_worker_0", "rollout_request"),
        ("tokenspeed_server_rollout_0", "tokenspeed_generate"),
    ]
    # The forwards that served r1, on TP0 only: the other TP ranks ran the same forwards.
    assert [(rank, name) for rank, name, _ in path[2:]] == [("TP0", "forward_batch")] * len(forwards)
    assert [round(ts - base_us, 3) for _, _, ts in path[2:]] == forwards
