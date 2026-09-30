import gzip
import json
import os
import socket

import pytest

from rl_trace_observer.context import STEP_MARKER_ATTRIBUTE
from rl_trace_observer.merger import build_manifest, merge_sources, read_source, select
from rl_trace_observer.merger.cli import main
from rl_trace_observer.merger.sources import TORCH
from rl_trace_observer.output import safe_component
from rl_trace_observer.process_record import SCHEMA_VERSION, record_path, register_artifact

BASE_NS = 1_790_000_000_000_000_000
MARKER = {STEP_MARKER_ATTRIBUTE: True}


def _record(root, host, pid, *, rank=None, actor=None, artifacts=(), run_id=None, role=None, lifetime_ns=None):
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
                "run_id": run_id,
                "role": role,
                **(
                    {"clock": {"wall_time_ns": lifetime_ns[0]}, "updated_wall_time_ns": lifetime_ns[1]}
                    if lifetime_ns
                    else {}
                ),
            }
        )
    )
    return path


def _jsonl(root, host, pid, lines=None):
    path = root / f"rl-insight-{host}-pid-{pid}.chrome.jsonl"
    span = {"name": "actor_update", "ph": "X", "ts": BASE_NS / 1000, "dur": 5.0, "pid": str(pid), "tid": "rank_0"}
    path.write_text("".join(line + "\n" for line in (lines or [json.dumps(span)])))
    return path


def _torch(root, pid, rank, *, distributed=True, step=1, base_ns=BASE_NS):
    path = root / f"actor_train_step{step}_rank{rank}-of-2_pid{pid}_20260930151150960.json.gz"
    data = {
        "baseTimeNanoseconds": base_ns,
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
    register_artifact(tmp_path, "viztracer", second, global_step=1)
    register_artifact(tmp_path, "viztracer", second, global_step=1)
    register_artifact(tmp_path, None, None, role="trainer")
    register_artifact(tmp_path, None, None, role="actor")

    record = json.loads(path.read_text())
    assert path == record_path(tmp_path)
    assert record["host"] == safe_component(socket.gethostname())
    assert record["os_pid"] == os.getpid()
    assert set(record["clock"]) == {"wall_time_ns", "monotonic_ns"}
    assert record["artifacts"] == [
        {"kind": "rl_insight_jsonl", "file": first.name},
        {"kind": "viztracer", "file": second.name, "global_step": 1},
    ]
    # The first role is kept; versions of the loaded frameworks are recorded.
    assert record["role"] == "trainer"
    assert "rl-trace-observer" in record["versions"]
    assert "run_id" in record
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


@pytest.mark.parametrize("torch_dir", ["a", "z"])
def test_rl_insight_file_without_a_record_never_claims_another_trace(tmp_path, torch_dir):
    # Without records the Torch trace's host is unknown, whichever file is read first.
    _jsonl(tmp_path, "node-1", 1234)
    (tmp_path / torch_dir).mkdir()
    _torch(tmp_path / torch_dir, 1234, rank=0)

    manifest, sources = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["unlinked"]
    assert sorted(str(source.process) for source in sources) == ["None", "node-1:1234"]


def test_artifact_that_cannot_be_read_after_hashing_is_incomplete(tmp_path, monkeypatch):
    from rl_trace_observer.merger import manifest as manifest_module

    _jsonl(tmp_path, "node-1", 1234)

    def vanish(kind, path):
        raise FileNotFoundError(2, "No such file or directory", str(path))

    monkeypatch.setattr(manifest_module, "read_source", vanish)
    manifest, sources = build_manifest([tmp_path])

    assert [problem.kind for problem in manifest.problems] == ["incomplete"]
    assert sources == []


def test_manifest_is_written_when_a_stale_trace_cannot_be_removed(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _torch(artifacts, 1234, rank=0)
    output = tmp_path / "merged.json"
    output.write_text("{}")
    unlink = type(output).unlink

    def refuse(path, missing_ok=False):
        if path == output:
            raise PermissionError(13, "Permission denied", str(path))
        return unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(type(output), "unlink", refuse)

    # --strict fails on the unlinked trace.
    assert main([str(artifacts), "-o", str(output), "--strict"]) == 1
    assert json.loads((tmp_path / "merged.manifest.json").read_text())["output"] is None


def test_register_artifact_through_a_symlinked_directory_updates_one_record(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)

    register_artifact(real, "rl_insight_jsonl", real / "rl-insight-a.chrome.jsonl")
    register_artifact(link, "viztracer", link / "step-1-role-e2e-rank-0-pid-1.viztracer.json")

    record = json.loads(record_path(real).read_text())
    assert [entry["file"] for entry in record["artifacts"]] == [
        "rl-insight-a.chrome.jsonl",
        "step-1-role-e2e-rank-0-pid-1.viztracer.json",
    ]


def _span_line(name, start_us, dur_us, pid, lane, **args):
    return json.dumps(
        {"name": name, "ph": "X", "ts": start_us, "dur": dur_us, "pid": str(pid), "tid": lane, "args": args}
    )


def _two_step_run(root, run_id="run-a"):
    """A trainer marking steps 1 and 2, an actor with spans and a Torch trace in each."""
    base_us = BASE_NS // 1000
    _record(root, "node-1", 10, run_id=run_id, role="trainer")
    _jsonl(
        root,
        "node-1",
        10,
        lines=[
            _span_line("global_step", base_us, 1000, 10, "trainer", **MARKER, global_step=1, run_id=run_id),
            _span_line("global_step", base_us + 2000, 1000, 10, "trainer", **MARKER, global_step=2, run_id=run_id),
        ],
    )
    _record(root, "node-1", 20, rank=0, run_id=run_id)
    _jsonl(
        root,
        "node-1",
        20,
        lines=[
            _span_line("actor_update", base_us + 100, 10, 20, "rank_0"),
            _span_line("actor_update", base_us + 2100, 10, 20, "rank_0"),
        ],
    )
    _torch(root, 20, rank=0, step=1)
    _torch(root, 20, rank=0, step=2, base_ns=BASE_NS + 2_000_000)


def test_sessions_are_the_steps_of_a_run(tmp_path):
    _two_step_run(tmp_path)

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources)

    assert manifest.runs == ["run-a"]
    assert manifest.problems == []
    assert len(kept) == len(sources) == 4
    assert [(session.id, session.start_time_ns, session.end_time_ns) for session in manifest.sessions] == [
        ("run-a-step-1", BASE_NS, BASE_NS + 1_000_000),
        ("run-a-step-2", BASE_NS + 2_000_000, BASE_NS + 3_000_000),
    ]
    assert [len(session.artifacts) for session in manifest.sessions] == [1, 1]
    torch = [artifact for artifact in manifest.artifacts if artifact.kind == "torch"]
    assert [(artifact.global_step, artifact.role, artifact.profile_session_id) for artifact in torch] == [
        (1, "actor_train", "run-a-step-1"),
        (2, "actor_train", "run-a-step-2"),
    ]
    assert manifest.processes["node-1:10@run-a"].role == "trainer"


def test_one_step_keeps_its_traces_and_the_spans_in_its_window(tmp_path):
    _two_step_run(tmp_path)

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, steps=[2])

    assert manifest.problems == []
    assert sorted(source.kind for source in kept) == ["rl_insight", "rl_insight", "torch"]
    assert [source.global_step for source in kept if source.kind == "torch"] == [2]
    timed = [(source.os_pid, event["name"]) for source in kept for event in source.events if "ts" in event]
    # Step 2's marker, the actor's step-2 span, and the step-2 Torch trace.
    assert sorted(timed) == [(10, "global_step"), (20, "actor_update"), (20, "actor_update"), (20, "gemm")]
    assert [
        event["args"]["global_step"] for source in kept for event in source.events if event["name"] == "global_step"
    ] == [2]
    result = merge_sources(kept, manifest.processes)
    assert result.global_base_ns == BASE_NS + 2_000_000
    assert [artifact.selected for artifact in manifest.artifacts if artifact.kind == "torch"] == [False, True]


def test_several_runs_must_be_chosen_between(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    _two_step_run(first, run_id="run-a")
    _record(second, "node-2", 30, rank=0, run_id="run-b")
    _jsonl(second, "node-2", 30)

    manifest, sources = build_manifest([tmp_path])
    select(manifest, sources)
    assert [problem.kind for problem in manifest.problems] == ["mixed_runs"]

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, run="run-b")
    assert manifest.problems == []
    assert [source.process for source in kept] == ["node-2:30@run-b"]


def test_a_step_without_a_window_leaves_rl_insight_spans_out(tmp_path):
    _record(tmp_path, "node-1", 20, rank=0, run_id="run-a")
    _jsonl(tmp_path, "node-1", 20)
    _torch(tmp_path, 20, rank=0, step=1)

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, steps=[1])

    assert [problem.kind for problem in manifest.problems] == ["no_step_window"]
    assert [source.kind for source in kept] == ["torch"]


def test_cli_merges_one_step_and_records_the_selection(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _two_step_run(artifacts)
    output = tmp_path / "step2.json"

    assert main([str(artifacts), "-o", str(output), "--strict", "--step", "2"]) == 0
    manifest = json.loads((tmp_path / "step2.manifest.json").read_text())
    assert manifest["selection"] == {"run": None, "steps": [2]}
    assert [session["global_step"] for session in manifest["sessions"]] == [1, 2]
    names = [event["name"] for event in json.loads(output.read_text())["traceEvents"] if event["ph"] == "X"]
    assert names.count("global_step") == 1

    assert main([str(artifacts), "-o", str(output), "--run", "run-z"]) == 1


def test_a_pid_reused_by_another_run_keeps_each_runs_identity(tmp_path):
    # The same container ran twice: same hostname and pid, different runs.
    first, second = tmp_path / "a", tmp_path / "b"
    for root, run_id in ((first, "run-a"), (second, "run-b")):
        root.mkdir()
        _record(root, "node-1", 1, run_id=run_id, artifacts=["rl-insight-node-1-pid-1.chrome.jsonl"])
        _jsonl(root, "node-1", 1, lines=[_span_line("actor_update", BASE_NS / 1000, 10, 1, "rank_0", run=run_id)])
        # An idle process of each run left the same empty file.
        _record(root, "node-1", 2, run_id=run_id, artifacts=["rl-insight-node-1-pid-2.chrome.jsonl"])
        (root / "rl-insight-node-1-pid-2.chrome.jsonl").touch()

    manifest, sources = build_manifest([tmp_path])

    assert manifest.problems == []
    assert manifest.runs == ["run-a", "run-b"]
    assert sorted(source.process for source in sources) == [
        "node-1:1@run-a",
        "node-1:1@run-b",
        "node-1:2@run-a",
        "node-1:2@run-b",
    ]
    kept = select(manifest, sources, run="run-b")
    assert sorted(source.process for source in kept) == ["node-1:1@run-b", "node-1:2@run-b"]


def test_malformed_step_markers_are_incomplete(tmp_path):
    _record(tmp_path, "node-1", 10, run_id="run-a")
    _jsonl(
        tmp_path,
        "node-1",
        10,
        lines=[_span_line("global_step", BASE_NS / 1000, 10, 10, "trainer", **MARKER, global_step="x")],
    )

    manifest, sources = build_manifest([tmp_path])
    select(manifest, sources)

    assert [(problem.kind, problem.detail) for problem in manifest.problems] == [
        ("incomplete", "1 malformed global_step spans")
    ]


def test_a_step_without_a_window_is_reported_without_rl_insight_sources(tmp_path):
    _record(tmp_path, "node-1", 20, rank=0, run_id="run-a")
    _torch(tmp_path, 20, rank=0, step=1)

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, steps=[1])

    assert [problem.kind for problem in manifest.problems] == ["no_step_window"]
    assert [source.kind for source in kept] == ["torch"]


def test_a_torch_trace_of_a_reused_pid_and_rank_links_to_the_run_it_ran_in(tmp_path):
    # Torch traces are not registered, so the records' lifetimes tell the runs apart.
    hour = 3_600_000_000_000
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _record(tmp_path / "a", "node-1", 7, rank=0, run_id="run-a", lifetime_ns=(BASE_NS - 2 * hour, BASE_NS - hour))
    _record(tmp_path / "b", "node-1", 7, rank=0, run_id="run-b", lifetime_ns=(BASE_NS - 1_000, BASE_NS + hour))
    _torch(tmp_path, 7, rank=0)

    manifest, sources = build_manifest([tmp_path])

    assert manifest.problems == []
    assert [source.process for source in sources] == ["node-1:7@run-b"]


def test_a_user_span_named_global_step_is_not_a_step_marker(tmp_path):
    _record(tmp_path, "node-1", 10, run_id="run-a")
    _jsonl(tmp_path, "node-1", 10, lines=[_span_line("global_step", BASE_NS / 1000, 10, 10, "app", step="warmup")])

    manifest, sources = build_manifest([tmp_path])
    select(manifest, sources)

    assert manifest.problems == []
    assert manifest.sessions == []


def test_torch_step_is_the_one_next_to_the_rank(tmp_path):
    path = _torch(tmp_path, 1234, rank=0, step=1)
    renamed = path.with_name("step99_" + path.name)
    path.rename(renamed)

    source = read_source(TORCH, renamed)

    assert (source.global_step, source.role) == (1, "step99_actor_train")


def test_select_can_be_called_again_for_another_step(tmp_path):
    _two_step_run(tmp_path)
    manifest, sources = build_manifest([tmp_path])

    def step_markers(kept):
        return [
            event["args"]["global_step"] for source in kept for event in source.events if event["name"] == "global_step"
        ]

    assert step_markers(select(manifest, sources, steps=[1])) == [1]
    assert step_markers(select(manifest, sources, steps=[2])) == [2]
    assert manifest.problems == []
    # Selection problems are replaced, not accumulated.
    select(manifest, sources, steps=[9])
    select(manifest, sources, steps=[9])
    assert [problem.kind for problem in manifest.problems] == ["no_step_window"]


def test_a_malformed_event_time_is_incomplete_when_selecting_steps(tmp_path):
    _two_step_run(tmp_path)
    path = tmp_path / "rl-insight-node-1-pid-20.chrome.jsonl"
    event = {"name": "bad", "ph": "X", "ts": BASE_NS // 1000, "dur": "long", "pid": "20", "tid": "rank_0"}
    path.write_text(path.read_text() + json.dumps(event) + "\n")

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, steps=[1])

    assert [(problem.kind, problem.detail) for problem in manifest.problems] == [
        ("incomplete", "1 events with a malformed time")
    ]
    assert "bad" not in {event["name"] for source in kept for event in source.events}


def test_step_markers_name_the_run_of_a_source_without_a_record(tmp_path):
    _jsonl(
        tmp_path,
        "node-1",
        10,
        lines=[_span_line("global_step", BASE_NS / 1000, 10, 10, "trainer", **MARKER, global_step=1, run_id="run-a")],
    )

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, run="run-a")

    assert [source.os_pid for source in kept] == [10]
    assert [session.id for session in manifest.sessions] == ["run-a-step-1"]


def test_runs_named_only_by_step_markers_are_listed_and_selectable(tmp_path):
    for pid, run_id in ((10, "run-a"), (11, "run-b")):
        _jsonl(
            tmp_path,
            "node-1",
            pid,
            lines=[
                _span_line("global_step", BASE_NS / 1000, 10, pid, "trainer", **MARKER, global_step=1, run_id=run_id)
            ],
        )

    manifest, sources = build_manifest([tmp_path])
    select(manifest, sources)
    assert manifest.runs == ["run-a", "run-b"]
    assert [problem.kind for problem in manifest.problems] == ["mixed_runs"]

    assert main([str(tmp_path), "-o", str(tmp_path / "out" / "a.json"), "--strict", "--run", "run-a"]) == 0
    written = json.loads((tmp_path / "out" / "a.manifest.json").read_text())
    assert [a["run_id"] for a in written["artifacts"] if a["selected"]] == ["run-a"]


def test_an_overflowing_marker_time_is_incomplete(tmp_path):
    marker = _span_line("global_step", BASE_NS / 1000, 10, 10, "trainer", **MARKER, global_step=1)
    _jsonl(tmp_path, "node-1", 10, lines=[marker.replace('"dur": 10', '"dur": 1e309')])

    manifest, sources = build_manifest([tmp_path])
    select(manifest, sources, steps=[1])

    # The span is neither a window nor a mergeable event; nothing raises.
    assert {problem.kind for problem in manifest.problems} == {"incomplete", "no_step_window"}


def test_artifacts_take_role_and_step_from_their_process_record(tmp_path):
    _two_step_run(tmp_path)
    # A registered trace whose filename carries no step.
    torch = _torch(tmp_path, 20, rank=0, step=1)
    unstepped = torch.with_name(torch.name.replace("_step1", ""))
    torch.rename(unstepped)
    record = tmp_path / "rl-trace-process-node-1-pid-20.json"
    data = json.loads(record.read_text())
    data["artifacts"].append({"kind": "torch", "file": unstepped.name, "global_step": 1})
    record.write_text(json.dumps(data))

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, steps=[1])

    roles = {artifact.path: artifact.role for artifact in manifest.artifacts}
    assert roles[str(tmp_path / "rl-insight-node-1-pid-10.chrome.jsonl")] == "trainer"
    assert str(unstepped) in {str(source.path) for source in kept}
