"""rl-trace-merge: the manifest is always written, and inputs are never touched."""

import json

import pytest
from trace_artifacts import BASE_NS, TracedProcess, two_step_run, write_torch

from rl_trace_observer.merger.cli import main


def _manifest(path):
    return json.loads(path.read_text())


def test_cli_writes_the_manifest_and_strict_fails_on_problems(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    worker = TracedProcess(artifacts, 1234, rank=0)
    worker.jsonl()
    output = tmp_path / "out" / "merged.json"

    assert main([str(artifacts), "-o", str(output), "--strict"]) == 0
    manifest = _manifest(tmp_path / "out" / "merged.manifest.json")
    assert (manifest["schema_version"], manifest["global_base_time_ns"]) == (1, BASE_NS)
    assert [process["key"] for process in manifest["processes"]] == [worker.key]
    assert [artifact["process"] for artifact in manifest["artifacts"]] == [worker.key]
    assert manifest["problems"] == []

    write_torch(artifacts / "actor_train_step1_rank1-of-2_pid4321_20260930151150960.json.gz", 4321)
    assert main([str(artifacts), "-o", str(output), "--strict"]) == 1
    # A failed run leaves its manifest but no trace from the earlier run.
    assert not output.exists()
    assert _manifest(tmp_path / "out" / "merged.manifest.json")["output"] is None
    assert main([str(artifacts), "-o", str(output)]) == 0
    assert output.exists()


def test_cli_merges_one_step_and_records_the_selection(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    two_step_run(artifacts)
    output = tmp_path / "step2.json"

    assert main([str(artifacts), "-o", str(output), "--strict", "--step", "2"]) == 0
    manifest = _manifest(tmp_path / "step2.manifest.json")
    assert manifest["selection"] == {"run": None, "steps": [2]}
    assert [session["global_step"] for session in manifest["sessions"]] == [1, 2]
    names = [event["name"] for event in json.loads(output.read_text())["traceEvents"] if event["ph"] == "X"]
    assert names.count("global_step") == 1

    assert main([str(artifacts), "-o", str(output), "--run", "run-z"]) == 1


def test_cli_creates_the_directory_of_an_explicit_manifest(tmp_path):
    TracedProcess(tmp_path, 1234).jsonl()
    manifest = tmp_path / "elsewhere" / "session.json"

    assert main([str(tmp_path), "-o", str(tmp_path / "merged.json"), "--manifest", str(manifest)]) == 0
    assert _manifest(manifest)["problems"] == []


def test_idle_processes_fail_cleanly(tmp_path):
    # Every client creates its JSONL up front, so idle processes leave identical empty files.
    for pid in (1, 2):
        TracedProcess(tmp_path, pid).jsonl([])
    output = tmp_path / "out" / "merged.json"

    assert main([str(tmp_path), "-o", str(output)]) == 1
    assert not output.exists()
    assert _manifest(tmp_path / "out" / "merged.manifest.json")["problems"] == []


def test_the_manifest_is_written_when_no_artifact_is_readable(tmp_path):
    TracedProcess(tmp_path, 1234).register(tmp_path / "rl-insight-node-1-pid-1234.chrome.jsonl", "rl_insight_jsonl")

    assert main([str(tmp_path), "-o", str(tmp_path / "merged.json")]) == 1
    assert [problem["kind"] for problem in _manifest(tmp_path / "merged.manifest.json")["problems"]] == ["missing"]


def test_the_manifest_names_no_output_when_the_trace_cannot_be_written(tmp_path):
    TracedProcess(tmp_path, 1234).jsonl()
    (tmp_path / "blocked").write_text("a file, not a directory")
    manifest = tmp_path / "session.json"

    assert main([str(tmp_path), "-o", str(tmp_path / "blocked" / "merged.json"), "--manifest", str(manifest)]) == 1
    assert _manifest(manifest)["output"] is None


def test_the_trace_is_removed_when_its_manifest_cannot_be_written(tmp_path):
    TracedProcess(tmp_path, 1234).jsonl()
    (tmp_path / "blocked").write_text("a file, not a directory")
    output = tmp_path / "merged.json"

    assert main([str(tmp_path), "-o", str(output), "--manifest", str(tmp_path / "blocked" / "m.json")]) == 1
    assert not output.exists()


def test_the_manifest_is_written_when_a_stale_trace_cannot_be_removed(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    TracedProcess(artifacts, 1234).torch(1, register=False)
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
    assert _manifest(tmp_path / "merged.manifest.json")["output"] is None


def test_the_cli_refuses_to_overwrite_or_remove_an_input(tmp_path):
    torch = TracedProcess(tmp_path, 1234).torch(1, register=False)
    content = torch.read_bytes()

    # --strict fails (unlinked trace), which would otherwise remove the "earlier" output.
    with pytest.raises(SystemExit):
        main([str(tmp_path), "-o", str(torch), "--strict"])
    with pytest.raises(SystemExit):
        main([str(tmp_path), "-o", str(tmp_path / "merged.json"), "--manifest", str(torch)])
    assert torch.read_bytes() == content


def test_the_cli_refuses_a_manifest_at_the_output_path(tmp_path):
    TracedProcess(tmp_path, 1234).jsonl()
    output = tmp_path / "out" / "merged.json"

    with pytest.raises(SystemExit):
        main([str(tmp_path), "-o", str(output), "--manifest", str(tmp_path / "out" / ".." / "out" / "merged.json")])
    assert not output.exists()
