import gzip
import json

import pytest

from rl_trace_observer.merger import discover, merge_sources, read_source
from rl_trace_observer.merger.cli import main
from rl_trace_observer.merger.sources import RL_INSIGHT, TORCH, VIZTRACER, classify

BASE_NS = 1_790_000_000_000_000_000
TORCH_NAME = "actor_train_step3_rank0-of-2_pid{pid}_20260930140841671.json.gz"


def _write_torch(path, events, base_ns=BASE_NS):
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump({"baseTimeNanoseconds": base_ns, "traceEvents": events}, file)
    return path


def _write_jsonl(path, lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def _span(name, ts_us, pid="1234", lane="rank_0"):
    return json.dumps({"name": name, "ph": "X", "ts": ts_us, "dur": 10.0, "pid": pid, "tid": lane, "args": {}})


def _cpu_events(pid, ts):
    return [
        {"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": "python3"}},
        {"ph": "M", "name": "thread_name", "pid": pid, "tid": pid, "args": {"name": f"thread {pid}"}},
        {"ph": "X", "name": "aten::mm", "pid": pid, "tid": pid, "ts": ts, "dur": 5.0},
        {"ph": "s", "name": "ac2g", "pid": pid, "tid": pid, "ts": ts, "id": 7, "cat": "ac2g"},
        {"ph": "f", "name": "ac2g", "pid": 0, "tid": 7, "ts": ts + 1, "id": 7, "cat": "ac2g", "bp": "e"},
    ]


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("rl-insight-node-1-pid-42.chrome.jsonl", RL_INSIGHT),
        ("actor_train_step3_rank0-of-8_tp0-pp0_pid42_20260930140841671.json.gz", TORCH),
        ("actor_update_rank1_pid42_20260930140841671_part1.json.gz", TORCH),
        ("step-3-role-e2e-rank-0-pid-42.viztracer.json", VIZTRACER),
        ("notes.json", None),
    ],
)
def test_classify(tmp_path, name, kind):
    assert classify(tmp_path / name) == kind


def test_discover_searches_directories_and_rejects_unknown_files(tmp_path):
    (tmp_path / "node-1").mkdir()
    jsonl = _write_jsonl(tmp_path / "node-1" / "rl-insight-node-1-pid-1.chrome.jsonl", [])
    (tmp_path / "README.txt").write_text("")

    assert discover([tmp_path]) == [(RL_INSIGHT, jsonl.resolve())]
    with pytest.raises(ValueError, match="Unrecognized"):
        discover([tmp_path / "README.txt"])


def test_rl_insight_jsonl_tolerates_only_a_truncated_final_line(tmp_path):
    path = _write_jsonl(tmp_path / "rl-insight-node-1-pid-1.chrome.jsonl", [_span("a", 1.0), '{"name": "b"'])
    assert [event["name"] for event in read_source(RL_INSIGHT, path).events] == ["a"]

    _write_jsonl(path, ['{"name": "b"', _span("a", 1.0)])
    with pytest.raises(ValueError, match="Corrupt line 1"):
        read_source(RL_INSIGHT, path)


def test_anchored_trace_requires_base_time(tmp_path):
    path = tmp_path / TORCH_NAME.format(pid=1)
    with gzip.open(path, "wt") as file:
        json.dump({"traceEvents": []}, file)

    with pytest.raises(ValueError, match="baseTimeNanoseconds"):
        read_source(TORCH, path)


def test_merge_aligns_sources_on_the_epoch_timeline(tmp_path):
    # RL-Insight ts is epoch µs; Torch ts is relative to baseTimeNanoseconds.
    jsonl = _write_jsonl(
        tmp_path / "rl-insight-node-1-pid-1234.chrome.jsonl", [_span("actor_update", BASE_NS / 1000 + 2)]
    )
    torch = _write_torch(tmp_path / TORCH_NAME.format(pid=1234), _cpu_events(1234, 5.0))

    result = merge_sources(read_source(kind, path) for kind, path in discover([tmp_path]))

    assert result.global_base_ns == BASE_NS + 2000
    timed = {event["name"]: event["ts"] for event in result.trace["traceEvents"] if event["ph"] == "X"}
    assert timed == {"actor_update": 0.0, "aten::mm": 3.0}
    assert jsonl.name in json.dumps(result.trace) and torch.name in json.dumps(result.trace)


def test_merge_isolates_processes_threads_and_flows_per_source(tmp_path):
    # Two nodes can produce the same OS pid, tid and Kineto flow id.
    _write_torch(tmp_path / TORCH_NAME.format(pid=1234), _cpu_events(1234, 0.0))
    other = tmp_path / "other"
    other.mkdir()
    _write_torch(other / TORCH_NAME.replace("rank0", "rank1").format(pid=1234), _cpu_events(1234, 0.0))

    result = merge_sources(read_source(kind, path) for kind, path in discover([tmp_path]))
    events = [event for event in result.trace["traceEvents"] if event["ph"] != "M"]

    mm_events = [event for event in events if event["name"] == "aten::mm"]
    assert len({event["pid"] for event in mm_events}) == 2
    assert len({event["tid"] for event in mm_events}) == 2
    thread_names = [e for e in result.trace["traceEvents"] if e["name"] == "thread_name"]
    assert len({event["tid"] for event in thread_names}) == len(thread_names)

    flow_ids = [
        [event["id"] for event in events if event["pid"] in pids and event["ph"] in "sf"]
        for pids in (
            {mm_events[0]["pid"], mm_events[0]["pid"] + 1},
            {mm_events[1]["pid"], mm_events[1]["pid"] + 1},
        )
    ]
    assert all(len(set(ids)) == 1 for ids in flow_ids)
    assert flow_ids[0][0] != flow_ids[1][0]


def test_merge_names_processes_and_threads(tmp_path):
    _write_jsonl(tmp_path / "rl-insight-node-1-pid-1234.chrome.jsonl", [_span("actor_update", 1.0)])
    _write_torch(tmp_path / TORCH_NAME.format(pid=1234), _cpu_events(1234, 0.0))

    result = merge_sources(read_source(kind, path) for kind, path in discover([tmp_path]))
    names = {
        (event["name"], event["args"]["name"])
        for event in result.trace["traceEvents"]
        if event["name"] in {"process_name", "thread_name"}
    }

    assert ("process_name", "RL-Insight node-1 pid 1234") in names
    assert ("thread_name", "rank_0") in names
    # The profiled process is named after its source; other pids keep their own name.
    assert ("process_name", "Torch actor_train_step3_rank0-of-2 pid 1234") in names
    assert ("process_name", "Torch actor_train_step3_rank0-of-2 pid 1234 · 0") in names
    assert ("thread_name", "thread 1234") in names
    assert all(isinstance(e["pid"], int) and isinstance(e["tid"], int) for e in result.trace["traceEvents"])


def test_negative_durations_are_dropped_with_a_warning(tmp_path):
    events = _cpu_events(1234, 0.0)
    events[2]["dur"] = -1.0
    _write_torch(tmp_path / TORCH_NAME.format(pid=1234), events)

    result = merge_sources(read_source(kind, path) for kind, path in discover([tmp_path]))

    assert "aten::mm" not in {event["name"] for event in result.trace["traceEvents"]}
    assert result.warnings and "dropped 1 events" in result.warnings[0]


def test_cli_writes_trace_and_strict_mode_fails_on_warnings(tmp_path):
    events = _cpu_events(1234, 0.0)
    _write_torch(tmp_path / TORCH_NAME.format(pid=1234), events)
    output = tmp_path / "out" / "merged.json"

    assert main([str(tmp_path), "-o", str(output)]) == 0
    assert json.loads(output.read_text())["traceEvents"]

    events[2]["dur"] = -1.0
    _write_torch(tmp_path / TORCH_NAME.format(pid=1234), events)
    assert main([str(tmp_path), "-o", str(output), "--strict"]) == 1


def test_cli_fails_without_artifacts(tmp_path):
    assert main([str(tmp_path), "-o", str(tmp_path / "merged.json")]) == 1


def test_kineto_bookkeeping_processes_are_dropped(tmp_path):
    events = _cpu_events(1234, 0.0) + [
        {"ph": "X", "name": "PyTorch Profiler (0)", "pid": "Spans", "tid": "PyTorch Profiler", "ts": 0.0, "dur": 9.0},
        {"ph": "i", "name": "Iteration Start: PyTorch Profiler", "pid": "Traces", "tid": "x", "ts": 0.0},
        {"ph": "i", "name": "Record Window End", "pid": "", "tid": "", "ts": 9.0},
    ]
    _write_torch(tmp_path / TORCH_NAME.format(pid=1234), events)

    result = merge_sources(read_source(kind, path) for kind, path in discover([tmp_path]))
    process_names = [e["args"]["name"] for e in result.trace["traceEvents"] if e["name"] == "process_name"]

    assert process_names == [
        "Torch actor_train_step3_rank0-of-2 pid 1234",
        "Torch actor_train_step3_rank0-of-2 pid 1234 · 0",
    ]


def test_viztracer_main_process_is_named_after_its_source(tmp_path):
    path = tmp_path / "step-3-role-e2e-rank-0-pid-42.viztracer.json"
    path.write_text(
        json.dumps(
            {
                "viztracer_metadata": {"baseTimeNanoseconds": BASE_NS},
                "traceEvents": [
                    {"ph": "M", "name": "process_name", "pid": 42, "tid": 42, "args": {"name": "MainProcess"}},
                    {"ph": "X", "name": "f", "pid": 42, "tid": 42, "ts": 1.0, "dur": 1.0},
                ],
            }
        )
    )

    result = merge_sources(read_source(kind, p) for kind, p in discover([tmp_path]))

    assert [e["args"]["name"] for e in result.trace["traceEvents"] if e["name"] == "process_name"] == [
        "VizTracer e2e step 3 rank 0 pid 42"
    ]
