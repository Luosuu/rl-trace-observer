import json
import os
import shutil
import socket

import pytest
from trace_artifacts import BASE_NS, BASE_US, TracedProcess, span, step_marker, two_step_run, write_torch

from rl_trace_observer.merger import build_manifest, merge_sources, select
from rl_trace_observer.merger import manifest as manifest_module
from rl_trace_observer.output import safe_component
from rl_trace_observer.process_record import record_path, register_artifact, set_process_role


def _kinds(manifest):
    return [problem.kind for problem in manifest.problems]


def _names(trace, kind):
    return {event["args"]["name"]: event["pid"] for event in trace["traceEvents"] if event["name"] == kind}


# Linking: an artifact belongs to the process that registered it.


def test_registered_artifacts_of_one_process_merge_into_one_perfetto_process(tmp_path):
    worker = TracedProcess(tmp_path, 1234, rank=0, actor="WorkerDict_0:0")
    worker.jsonl()
    worker.torch(1)

    manifest, sources = build_manifest([tmp_path])
    result = merge_sources(sources, manifest.processes)

    assert manifest.problems == []
    assert {artifact.process for artifact in manifest.artifacts} == {worker.key}
    processes, threads = _names(result.trace, "process_name"), _names(result.trace, "thread_name")
    # RL-Insight lanes and the Torch CPU thread share one process; the GPU device stays separate.
    title = "node-1.example pid 1234 · WorkerDict_0:0"
    assert set(processes) == {title, f"{title} · 0"}
    assert threads["RL-Insight · rank_0"] == threads["Torch · 1234"] == processes[title]


def test_unregistered_artifacts_are_unlinked_and_merged_on_their_own(tmp_path):
    # e.g. a run without this package: nothing registered these files.
    jsonl = TracedProcess(tmp_path, 1234).jsonl(register=False)
    torch = write_torch(tmp_path / "actor_train_step1_rank0-of-2_pid1234_20260930151150960.json.gz", 1234)
    (tmp_path / "rl-trace-process-node-1-pid-1234.json").unlink()

    manifest, sources = build_manifest([tmp_path])
    result = merge_sources(sources, manifest.processes)

    assert [(problem.kind, problem.path) for problem in manifest.problems] == [
        ("unlinked", str(torch)),
        ("unlinked", str(jsonl)),
    ]
    assert "RL-Insight node-1 pid 1234" in _names(result.trace, "process_name")


def test_a_pid_reused_by_another_run_keeps_each_runs_identity(tmp_path):
    # The same container ran twice: same hostname and pid, different runs.
    keys = []
    for run_id in ("run-a", "run-b"):
        (tmp_path / run_id).mkdir()
        busy = TracedProcess(tmp_path / run_id, 1, run_id=run_id)
        busy.jsonl([span("actor_update", pid=1, run=run_id)])
        # An idle process of each run leaves the same empty file; it is not a duplicate.
        TracedProcess(tmp_path / run_id, 2, run_id=run_id).jsonl([])
        keys.append(busy.key)

    manifest, sources = build_manifest([tmp_path])

    assert manifest.problems == []
    assert manifest.runs == ["run-a", "run-b"]
    assert {source.process for source in sources} >= set(keys)
    assert {source.process for source in select(manifest, sources, run="run-b")} == {"node-1:1@run-b", "node-1:2@run-b"}


def test_a_copied_output_directory_still_links(tmp_path):
    # Record entries are relative to the record, so collecting the directory elsewhere keeps them.
    (tmp_path / "node").mkdir()
    worker = TracedProcess(tmp_path / "node", 1234)
    worker.jsonl()
    shutil.copytree(tmp_path / "node", tmp_path / "collected")
    shutil.rmtree(tmp_path / "node")

    manifest, sources = build_manifest([tmp_path / "collected"])

    assert manifest.problems == []
    assert [source.process for source in sources] == [worker.key]


def test_artifacts_take_step_and_role_from_the_record(tmp_path):
    trainer, actor = two_step_run(tmp_path)

    manifest, _ = build_manifest([tmp_path])

    artifacts = {artifact.path: artifact for artifact in manifest.artifacts}
    trainer_jsonl = artifacts[str(tmp_path / "rl-insight-node-1-pid-10.chrome.jsonl")]
    # The trainer's JSONL has no role of its own and takes the process's.
    assert (trainer_jsonl.role, trainer_jsonl.global_step) == ("trainer", None)
    torch = sorted((a.global_step, a.role, a.profile_session_id) for a in manifest.artifacts if a.kind == "torch")
    assert torch == [(1, "actor_train", "run-a-step-1"), (2, "actor_train", "run-a-step-2")]


# Problems: reported, never raised.


def test_a_copy_of_an_artifact_is_a_duplicate_and_merged_once(tmp_path):
    original = TracedProcess(tmp_path, 1234).jsonl()
    (tmp_path / "copy").mkdir()
    shutil.copy(original, tmp_path / "copy" / original.name)

    manifest, sources = build_manifest([tmp_path])

    assert _kinds(manifest) == ["duplicate"]
    assert len(sources) == 1


def test_a_truncated_jsonl_is_incomplete_but_kept(tmp_path):
    TracedProcess(tmp_path, 1234).jsonl([span("a"), '{"name": "b"'])

    manifest, sources = build_manifest([tmp_path])

    assert [(problem.kind, problem.detail) for problem in manifest.problems] == [("incomplete", "final line truncated")]
    assert manifest.artifacts[0].complete is False
    assert len(sources) == 1


@pytest.mark.parametrize(
    "damage",
    [
        lambda path: path.write_bytes(path.read_bytes()[:40]),  # cut short
        lambda path: path.write_text('{"baseTimeNanoseconds": "soon", "traceEvents": []}'),  # malformed
    ],
)
def test_an_unreadable_torch_trace_is_incomplete_and_skipped(tmp_path, damage):
    damage(TracedProcess(tmp_path, 1234).torch(1))

    manifest, sources = build_manifest([tmp_path])

    assert _kinds(manifest) == ["incomplete"]
    assert sources == []


def test_a_null_jsonl_line_is_incomplete(tmp_path):
    TracedProcess(tmp_path, 1234).jsonl(["null"])

    manifest, sources = build_manifest([tmp_path])

    assert _kinds(manifest) == ["incomplete"]
    assert sources == []


@pytest.mark.parametrize("step", ["_sha256", "read_source"])
def test_a_file_that_cannot_be_read_is_incomplete(tmp_path, monkeypatch, step):
    TracedProcess(tmp_path, 1234).jsonl()

    def vanish(*args):
        raise FileNotFoundError(2, "No such file or directory")

    monkeypatch.setattr(manifest_module, step, vanish)
    manifest, sources = build_manifest([tmp_path])

    assert _kinds(manifest) == ["incomplete"]
    assert manifest.artifacts[0].complete is False
    assert sources == []


def test_a_registered_artifact_that_never_arrived_is_missing(tmp_path):
    process = TracedProcess(tmp_path, 1234)
    process.register(tmp_path / "rl-insight-node-1-pid-1234.chrome.jsonl", "rl_insight_jsonl")

    manifest, _ = build_manifest([tmp_path])

    assert _kinds(manifest) == ["missing"]
    assert manifest.problems[0].path.endswith("rl-insight-node-1-pid-1234.chrome.jsonl")


@pytest.mark.parametrize(
    "content",
    [
        "{",
        "[1]",
        '{"schema_version": 1, "host": "node-1"}',
        '{"host": "node-1", "os_pid": 1234}',
        '{"schema_version": 2, "host": "node-1", "os_pid": 1234}',
        '{"schema_version": 1, "host": "node-1", "os_pid": 1234, "artifacts": [{"kind": "viztracer"}]}',
        '{"schema_version": 1, "host": "node-1", "os_pid": 1234, "artifacts": [{"file": "x", "global_step": "1"}]}',
        '{"schema_version": 1, "host": "node-1", "os_pid": 1234, "ray": "not-a-dict"}',
        '{"schema_version": 1, "host": "node-1", "os_pid": 1234, "clock": [1]}',
        '{"schema_version": 1, "host": "node-1", "os_pid": 1234, "run_id": ["a"]}',
        '{"schema_version": 1, "host": "node-1", "os_pid": 1234, "role": 7}',
    ],
)
def test_a_corrupt_record_is_incomplete(tmp_path, content):
    (tmp_path / "rl-trace-process-node-1-pid-1234.json").write_text(content)

    manifest, _ = build_manifest([tmp_path])

    assert _kinds(manifest) == ["incomplete"]


# Process records.


def test_register_artifact_writes_this_process_record(tmp_path):
    first = tmp_path / "rl-insight-a.chrome.jsonl"
    second = tmp_path / "step-1-role-e2e-rank-0-pid-1.viztracer.json"

    path = register_artifact(tmp_path, "rl_insight_jsonl", first)
    register_artifact(tmp_path, "viztracer", second, global_step=1, role="e2e")
    register_artifact(tmp_path, "viztracer", second, global_step=1, role="e2e")
    set_process_role(tmp_path, "trainer")
    set_process_role(tmp_path, "actor")

    record = json.loads(path.read_text())
    assert path == record_path(tmp_path)
    assert (record["host"], record["os_pid"]) == (safe_component(socket.gethostname()), os.getpid())
    assert set(record["clock"]) == {"wall_time_ns", "monotonic_ns"}
    assert record["artifacts"] == [
        {"kind": "rl_insight_jsonl", "file": first.name},
        {"kind": "viztracer", "file": second.name, "global_step": 1, "role": "e2e"},
    ]
    # The first role is kept; versions of the loaded frameworks are recorded.
    assert record["role"] == "trainer"
    assert "rl-trace-observer" in record["versions"]
    assert "run_id" in record
    assert not list(tmp_path.glob(".*.tmp"))


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


# Runs and profiled steps.


def test_sessions_are_the_steps_of_a_run(tmp_path):
    two_step_run(tmp_path)

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


def test_one_step_keeps_its_traces_and_the_spans_in_its_window(tmp_path):
    two_step_run(tmp_path)

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, steps=[2])

    assert manifest.problems == []
    timed = sorted((source.os_pid, event["name"]) for source in kept for event in source.events if "ts" in event)
    # Step 2's marker, the actor's step-2 span, and the step-2 Torch trace.
    assert timed == [(10, "global_step"), (20, "actor_update"), (20, "actor_update"), (20, "gemm")]
    assert merge_sources(kept, manifest.processes).global_base_ns == BASE_NS + 2_000_000
    assert [artifact.selected for artifact in manifest.artifacts if artifact.kind == "torch"] == [False, True]


def test_select_can_be_called_again_for_another_step(tmp_path):
    two_step_run(tmp_path)
    manifest, sources = build_manifest([tmp_path])

    def markers(kept):
        return [
            event["args"]["global_step"] for source in kept for event in source.events if event["name"] == "global_step"
        ]

    assert markers(select(manifest, sources, steps=[1])) == [1]
    assert markers(select(manifest, sources, steps=[2])) == [2]
    assert manifest.problems == []
    # Selection problems are replaced, not accumulated.
    select(manifest, sources, steps=[9])
    select(manifest, sources, steps=[9])
    assert _kinds(manifest) == ["no_step_window"]


def test_several_runs_must_be_chosen_between(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    two_step_run(tmp_path / "a", run_id="run-a")
    other = TracedProcess(tmp_path / "b", 30, host="node-2", run_id="run-b")
    other.jsonl()

    manifest, sources = build_manifest([tmp_path])
    select(manifest, sources)
    assert _kinds(manifest) == ["mixed_runs"]

    kept = select(manifest, sources, run="run-b")
    assert manifest.problems == []
    assert [source.process for source in kept] == [other.key]


@pytest.mark.parametrize("with_rl_insight", [True, False])
def test_a_step_without_a_window_leaves_rl_insight_spans_out(tmp_path, with_rl_insight):
    actor = TracedProcess(tmp_path, 20, rank=0)
    if with_rl_insight:
        actor.jsonl()
    actor.torch(1)

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, steps=[1])

    assert _kinds(manifest) == ["no_step_window"]
    assert [source.kind for source in kept] == ["torch"]


@pytest.mark.parametrize(
    "marker",
    [
        step_marker("x"),  # non-numeric step
        step_marker(1).replace('"dur": 1000', '"dur": 1e309'),  # overflows a float
        step_marker(1, run_id=["a"]),  # non-string run id
    ],
)
def test_a_malformed_step_marker_is_incomplete(tmp_path, marker):
    TracedProcess(tmp_path, 10, role="trainer").jsonl([marker])

    manifest, sources = build_manifest([tmp_path])
    select(manifest, sources)

    assert _kinds(manifest) == ["incomplete"]
    assert manifest.sessions == []


def test_a_user_span_named_global_step_is_not_a_step_marker(tmp_path):
    TracedProcess(tmp_path, 10).jsonl([span("global_step", lane="app", step="warmup")])

    manifest, sources = build_manifest([tmp_path])
    select(manifest, sources)

    assert manifest.problems == []
    assert manifest.sessions == []


def test_a_malformed_event_time_is_incomplete_when_selecting_steps(tmp_path):
    two_step_run(tmp_path)
    path = tmp_path / "rl-insight-node-1-pid-20.chrome.jsonl"
    path.write_text(path.read_text() + span("bad", BASE_US, "long", pid=20) + "\n")

    manifest, sources = build_manifest([tmp_path])
    kept = select(manifest, sources, steps=[1])

    assert [(problem.kind, problem.detail) for problem in manifest.problems] == [
        ("incomplete", "1 events with a malformed time")
    ]
    assert "bad" not in {event["name"] for source in kept for event in source.events}
