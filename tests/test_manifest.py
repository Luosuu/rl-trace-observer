import gzip
import json
import os
import socket

import pytest

from rl_trace_observer.merger import build_manifest, merge_sources
from rl_trace_observer.merger.cli import main
from rl_trace_observer.output import safe_component
from rl_trace_observer.process_record import SCHEMA_VERSION, record_path, register_artifact

BASE_NS = 1_790_000_000_000_000_000


def _record(root, host, pid, *, rank=None, actor=None, artifacts=()):
    path = root / f"rl-trace-process-{host}-pid-{pid}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "hostname": f"{host}.example",
                "host": host,
                "os_pid": pid,
                "ray": {"actor_name": actor} if actor else None,
                "torch_distributed": {"rank": rank, "world_size": 2} if rank is not None else None,
                "artifacts": [{"kind": "rl_insight_jsonl", "file": name} for name in artifacts],
            }
        )
    )
    return path


def _jsonl(root, host, pid, lines=None):
    path = root / f"rl-insight-{host}-pid-{pid}.chrome.jsonl"
    span = {"name": "actor_update", "ph": "X", "ts": BASE_NS / 1000, "dur": 5.0, "pid": str(pid), "tid": "rank_0"}
    path.write_text("".join(line + "\n" for line in (lines or [json.dumps(span)])))
    return path


def _torch(root, pid, rank, *, distributed=True):
    path = root / f"actor_train_step1_rank{rank}-of-2_pid{pid}_20260930151150960.json.gz"
    data = {
        "baseTimeNanoseconds": BASE_NS,
        "traceEvents": [
            {"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": "ray::WorkerDict"}},
            {"ph": "X", "name": "actor_update", "pid": pid, "tid": pid, "ts": 1.0, "dur": 3.0},
            {"ph": "X", "name": "gemm", "pid": 0, "tid": 7, "ts": 2.0, "dur": 1.0},
        ],
    }
    if distributed:
        data["distributedInfo"] = {"rank": rank, "world_size": 2}
    with gzip.open(path, "wt") as file:
        json.dump(data, file)
    return path


def _names(trace, kind):
    return {event["args"]["name"]: event["pid"] for event in trace["traceEvents"] if event["name"] == kind}


def test_process_record_links_torch_trace_to_the_same_process(tmp_path):
    _record(
        tmp_path, "node-1", 1234, rank=0, actor="WorkerDict_0:0", artifacts=["rl-insight-node-1-pid-1234.chrome.jsonl"]
    )
    _jsonl(tmp_path, "node-1", 1234)
    _torch(tmp_path, 1234, rank=0)

    manifest, sources = build_manifest([tmp_path])
    result = merge_sources(sources, manifest.processes)

    assert manifest.problems == []
    assert {artifact.process for artifact in manifest.artifacts} == {"node-1:1234"}
    processes = _names(result.trace, "process_name")
    threads = _names(result.trace, "thread_name")
    # RL-Insight lanes and the Torch CPU thread share one process; the GPU device stays separate.
    assert set(processes) == {
        "node-1.example pid 1234 · WorkerDict_0:0",
        "node-1.example pid 1234 · WorkerDict_0:0 · 0",
    }
    own = processes["node-1.example pid 1234 · WorkerDict_0:0"]
    assert threads["RL-Insight · rank_0"] == own
    assert threads["Torch · 1234"] == own


def test_rank_disambiguates_a_pid_reused_on_two_hosts(tmp_path):
    _record(tmp_path, "node-1", 1234, rank=0)
    _record(tmp_path, "node-2", 1234, rank=1)
    _torch(tmp_path, 1234, rank=1)

    manifest, sources = build_manifest([tmp_path])

    assert manifest.problems == []
    assert sources[0].process == "node-2:1234"


def test_pid_on_two_hosts_without_rank_is_ambiguous(tmp_path):
    _record(tmp_path, "node-1", 1234)
    _record(tmp_path, "node-2", 1234)
    torch = _torch(tmp_path, 1234, rank=0, distributed=False)
    torch.rename(tmp_path / "actor_train_step1_pid1234_20260930151150960.json.gz")

    manifest, sources = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["ambiguous"]
    assert sources[0].process is None


def test_torch_trace_without_a_record_is_unlinked_but_still_merged(tmp_path):
    _torch(tmp_path, 1234, rank=0)

    manifest, sources = build_manifest([tmp_path])
    result = merge_sources(sources, manifest.processes)

    assert [problem.kind for problem in manifest.problems] == ["unlinked"]
    assert "Torch actor_train_step1_rank0-of-2 pid 1234" in _names(result.trace, "process_name")


def test_rl_insight_without_records_still_names_its_process(tmp_path):
    _jsonl(tmp_path, "node-1", 1234)

    manifest, sources = build_manifest([tmp_path])
    result = merge_sources(sources, manifest.processes)

    assert manifest.problems == []
    assert set(_names(result.trace, "process_name")) == {"node-1 pid 1234"}


def test_duplicate_content_is_reported_and_merged_once(tmp_path):
    original = _jsonl(tmp_path, "node-1", 1234)
    (tmp_path / "copy").mkdir()
    (tmp_path / "copy" / original.name).write_bytes(original.read_bytes())

    manifest, sources = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["duplicate"]
    assert len(sources) == 1


def test_truncated_jsonl_is_incomplete_but_kept(tmp_path):
    span = json.dumps({"name": "a", "ph": "X", "ts": BASE_NS / 1000, "dur": 1.0, "pid": "1", "tid": "x"})
    _jsonl(tmp_path, "node-1", 1234, lines=[span, '{"name": "b"'])

    manifest, sources = build_manifest([tmp_path])

    assert [(problem.kind, problem.detail) for problem in manifest.problems] == [("incomplete", "final line truncated")]
    assert manifest.artifacts[0].complete is False
    assert len(sources) == 1


def test_unreadable_trace_is_incomplete_and_skipped(tmp_path):
    torch = _torch(tmp_path, 1234, rank=0)
    torch.write_bytes(torch.read_bytes()[:40])

    manifest, sources = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == [
        "incomplete",
    ]
    assert sources == []


def test_registered_artifact_that_never_arrived_is_missing(tmp_path):
    _record(tmp_path, "node-1", 1234, artifacts=["rl-insight-node-1-pid-1234.chrome.jsonl"])

    manifest, _ = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["missing"]
    assert manifest.problems[0].path.endswith("rl-insight-node-1-pid-1234.chrome.jsonl")


def test_cli_writes_manifest_and_strict_fails_on_problems(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _record(artifacts, "node-1", 1234, rank=0, artifacts=["rl-insight-node-1-pid-1234.chrome.jsonl"])
    _jsonl(artifacts, "node-1", 1234)
    output = tmp_path / "out" / "merged.json"

    assert main([str(artifacts), "-o", str(output), "--strict"]) == 0
    manifest = json.loads((tmp_path / "out" / "merged.manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["global_base_time_ns"] == BASE_NS
    assert [process["key"] for process in manifest["processes"]] == ["node-1:1234"]
    assert [artifact["process"] for artifact in manifest["artifacts"]] == ["node-1:1234"]
    assert manifest["problems"] == []

    _torch(artifacts, 4321, rank=1)
    assert main([str(artifacts), "-o", str(output), "--strict"]) == 1
    # A failed run leaves its manifest but no trace from the earlier run.
    assert not output.exists()
    assert json.loads((tmp_path / "out" / "merged.manifest.json").read_text())["output"] is None
    assert main([str(artifacts), "-o", str(output)]) == 0
    assert output.exists()


def test_cli_creates_the_directory_of_an_explicit_manifest(tmp_path):
    _jsonl(tmp_path, "node-1", 1234)
    manifest = tmp_path / "elsewhere" / "session.json"

    assert main([str(tmp_path), "-o", str(tmp_path / "merged.json"), "--manifest", str(manifest)]) == 0
    assert json.loads(manifest.read_text())["problems"] == []


def test_processes_without_spans_are_not_duplicates_and_fail_cleanly(tmp_path):
    # Every client creates its JSONL up front, so idle processes leave identical empty files.
    for pid in (1, 2):
        _record(tmp_path, "node-1", pid, artifacts=[f"rl-insight-node-1-pid-{pid}.chrome.jsonl"])
        (tmp_path / f"rl-insight-node-1-pid-{pid}.chrome.jsonl").touch()
    output = tmp_path / "out" / "merged.json"

    manifest, sources = build_manifest([tmp_path])
    assert manifest.problems == []
    assert len(sources) == 2

    assert main([str(tmp_path), "-o", str(output)]) == 1
    assert not output.exists()
    assert json.loads((tmp_path / "out" / "merged.manifest.json").read_text())["problems"] == []


def test_manifest_is_written_when_no_artifact_is_readable(tmp_path):
    _record(tmp_path, "node-1", 1234, artifacts=["rl-insight-node-1-pid-1234.chrome.jsonl"])
    output = tmp_path / "merged.json"

    assert main([str(tmp_path), "-o", str(output)]) == 1
    problems = json.loads((tmp_path / "merged.manifest.json").read_text())["problems"]
    assert [problem["kind"] for problem in problems] == ["missing"]


def test_artifact_that_cannot_be_hashed_is_incomplete(tmp_path, monkeypatch):
    from rl_trace_observer.merger import manifest as manifest_module

    _jsonl(tmp_path, "node-1", 1234)

    def fail(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(manifest_module, "_sha256", fail)
    manifest, sources = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["incomplete"]
    assert manifest.artifacts[0].complete is False
    assert sources == []


def test_register_artifact_writes_this_process_record(tmp_path):
    first = tmp_path / "rl-insight-a.chrome.jsonl"
    second = tmp_path / "step-1-role-e2e-rank-0-pid-1.viztracer.json"

    path = register_artifact(tmp_path, "rl_insight_jsonl", first)
    register_artifact(tmp_path, "viztracer", second)
    register_artifact(tmp_path, "viztracer", second)

    record = json.loads(path.read_text())
    assert path == record_path(tmp_path)
    assert record["host"] == safe_component(socket.gethostname())
    assert record["os_pid"] == os.getpid()
    assert set(record["clock"]) == {"wall_time_ns", "monotonic_ns"}
    assert record["artifacts"] == [
        {"kind": "rl_insight_jsonl", "file": first.name},
        {"kind": "viztracer", "file": second.name},
    ]
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "content",
    [
        "{",
        '{"host": "node-1"}',
        '{"host": "node-1", "os_pid": 1234, "artifacts": [{"kind": "viztracer"}]}',
        '{"schema_version": 1, "host": "node-1", "os_pid": 1234, "ray": "not-a-dict"}',
        '{"schema_version": 1, "host": "node-1", "os_pid": 1234, "artifacts": [{"kind": "viztracer"}]}',
        # Missing or unsupported versions are not read with version 1 semantics.
        '{"host": "node-1", "os_pid": 1234}',
        '{"schema_version": 2, "host": "node-1", "os_pid": 1234}',
    ],
)
def test_corrupt_record_is_incomplete(tmp_path, content):
    (tmp_path / "rl-trace-process-node-1-pid-1234.json").write_text(content)

    manifest, _ = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["incomplete"]


def test_sole_record_with_another_rank_does_not_claim_the_trace(tmp_path):
    # The rank-1 process's record is missing; the rank-0 record with the same pid is on another host.
    _record(tmp_path, "node-1", 1234, rank=0)
    _torch(tmp_path, 1234, rank=1)

    manifest, sources = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["unlinked"]
    assert sources[0].process is None


@pytest.mark.parametrize(
    "write",
    [
        lambda root: _jsonl(root, "node-1", 1234, lines=["null"]),
        lambda root: _torch(root, 1234, rank=0).write_bytes(
            gzip.compress(json.dumps({"baseTimeNanoseconds": "soon", "traceEvents": []}).encode())
        ),
    ],
)
def test_malformed_fields_make_an_artifact_incomplete(tmp_path, write):
    write(tmp_path)

    manifest, sources = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["incomplete"]
    assert sources == []


def test_manifest_names_no_output_when_the_trace_cannot_be_written(tmp_path):
    _jsonl(tmp_path, "node-1", 1234)
    (tmp_path / "blocked").write_text("a file, not a directory")
    manifest = tmp_path / "session.json"

    assert main([str(tmp_path), "-o", str(tmp_path / "blocked" / "merged.json"), "--manifest", str(manifest)]) == 1
    assert json.loads(manifest.read_text())["output"] is None


def test_records_without_rank_are_ambiguous_for_a_ranked_trace(tmp_path):
    # Records written before torch.distributed was initialized have no rank.
    _record(tmp_path, "node-1", 1234)
    _record(tmp_path, "node-2", 1234)
    _torch(tmp_path, 1234, rank=1)

    manifest, _ = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["ambiguous"]


def test_cli_refuses_to_overwrite_or_remove_an_input(tmp_path):
    torch = _torch(tmp_path, 1234, rank=0)
    content = torch.read_bytes()

    # --strict fails (unlinked trace), which would otherwise remove the "earlier" output.
    with pytest.raises(SystemExit):
        main([str(tmp_path), "-o", str(torch), "--strict"])
    with pytest.raises(SystemExit):
        main([str(tmp_path), "-o", str(tmp_path / "merged.json"), "--manifest", str(torch)])
    assert torch.read_bytes() == content


def test_cli_refuses_a_manifest_at_the_output_path(tmp_path):
    _jsonl(tmp_path, "node-1", 1234)
    output = tmp_path / "out" / "merged.json"

    with pytest.raises(SystemExit):
        main([str(tmp_path), "-o", str(output), "--manifest", str(tmp_path / "out" / ".." / "out" / "merged.json")])
    assert not output.exists()


def test_trace_is_removed_when_its_manifest_cannot_be_written(tmp_path):
    _jsonl(tmp_path, "node-1", 1234)
    (tmp_path / "blocked").write_text("a file, not a directory")
    output = tmp_path / "merged.json"

    assert main([str(tmp_path), "-o", str(output), "--manifest", str(tmp_path / "blocked" / "m.json")]) == 1
    assert not output.exists()
