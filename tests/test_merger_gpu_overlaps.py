"""Kernels that overlap on one Proton GPU stream go to overlap lanes, so every slice nests in Perfetto."""

import json

from trace_artifacts import write_tokenspeed_pair

from rl_trace_observer.merger import discover, merge_sources, read_source


def _write(tmp_path):
    """A Proton stream whose kernels overlap like programmatic dependent launches, each launched by a flow."""
    _, proton = write_tokenspeed_pair(tmp_path, "run-a-step-1")
    trace = json.loads(proton.read_text())
    offset = 2.0  # write_tokenspeed_pair's Proton anchor offset, µs
    # The fixture's flash_attn_kernel runs at 20-26 µs on lane 0.
    kernels = [
        ("nested_kernel", 21.0, 1.0),  # inside flash_attn_kernel
        ("rmsnorm_kernel", 25.0, 3.0),  # starts 1 µs before flash_attn_kernel ends
        ("next_kernel", 27.0, 2.0),  # after flash_attn_kernel: back on the stream
    ]
    for index, (name, start, dur) in enumerate(kernels, start=2):
        trace["traceEvents"] += [
            {"ph": "X", "name": name, "pid": 0, "tid": 2, "ts": start - offset, "dur": dur},
            {"ph": "s", "name": "launch", "pid": 0, "tid": 1, "ts": 13.0 + index * 0.1 - offset, "id": index},
            {"ph": "f", "name": "launch", "pid": 0, "tid": 2, "ts": start - offset, "id": index, "bp": "e"},
        ]
    proton.write_text(json.dumps(trace))
    merged = tmp_path / "merged.json"
    result = merge_sources(read_source(kind, path) for kind, path in discover([tmp_path]))
    merged.write_text(json.dumps(result.trace))
    return merged


def test_overlapping_kernels_get_overlap_lanes_with_their_launch_flows(tmp_path, load_in_perfetto):
    processor = load_in_perfetto(_write(tmp_path))

    def query(sql):
        return processor.query(sql).as_pandas_dataframe()

    spilled = query("select value from stats where name = 'slice_spill_overlapping_complete_event'")
    assert spilled.value[0] == 0
    lanes = query(
        """
        select s.name, t.name as thread from slice s join thread_track tt on s.track_id = tt.id
        join thread t using (utid) where t.name like 'Proton · GPU Stream%'
        """
    )
    assert dict(zip(lanes.name, lanes.thread, strict=True)) == {
        "flash_attn_kernel": "Proton · GPU Stream 7",
        "next_kernel": "Proton · GPU Stream 7",
        "nested_kernel": "Proton · GPU Stream 7 · overlap",
        "rmsnorm_kernel": "Proton · GPU Stream 7 · overlap",
    }
    # Every launch still ends on its own kernel.
    flows = query(
        """
        select i.name from flow f join slice o on f.slice_out = o.id join slice i on f.slice_in = i.id
        where o.name = 'attention'
        """
    )
    assert sorted(flows.name) == ["flash_attn_kernel", "nested_kernel", "next_kernel", "rmsnorm_kernel"]


def test_a_stream_without_overlaps_is_left_alone(tmp_path):
    write_tokenspeed_pair(tmp_path, "run-a-step-1")
    events = merge_sources(read_source(kind, path) for kind, path in discover([tmp_path])).trace["traceEvents"]
    threads = [e["args"]["name"] for e in events if e.get("name") == "thread_name"]
    assert not any("overlap" in name for name in threads)


def test_overlapping_kernels_on_a_kineto_stream_get_an_overlap_lane(tmp_path):
    import gzip

    from trace_artifacts import write_torch

    path = write_torch(tmp_path / "actor_rank0_pid10_20261011101626000.json.gz", 10)
    trace = json.loads(gzip.open(path, "rt").read())
    trace["traceEvents"] += [
        {"ph": "M", "name": "thread_name", "pid": 0, "tid": 7, "args": {"name": "stream 7 "}},
        # A split-K GEMM's reduce starts before the GEMM ends.
        {"ph": "X", "name": "splitk_reduce", "pid": 0, "tid": 7, "ts": 2.5, "dur": 1.0},
    ]
    with gzip.open(path, "wt") as file:
        json.dump(trace, file)

    events = merge_sources([read_source("torch", path)]).trace["traceEvents"]
    threads = {e["tid"]: e["args"]["name"] for e in events if e.get("name") == "thread_name"}
    lanes = {e["name"]: threads[e["tid"]] for e in events if e.get("ph") == "X" and e["name"] != "actor_update"}
    assert lanes == {"gemm": "Torch · stream 7 ", "splitk_reduce": "Torch · stream 7 · overlap"}
