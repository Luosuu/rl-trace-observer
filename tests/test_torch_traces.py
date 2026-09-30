import json
import types

from rl_trace_observer.integrations.verl import torch_traces
from rl_trace_observer.process_record import record_path


class _Profiler:
    def __init__(self):
        self.exported = []

    def export_chrome_trace(self, path):
        self.exported.append(path)


def _module():
    def get_torch_profiler(contents, save_path, role=None, save_file_prefix=None, rank=0, profile_step=None):
        return _Profiler()

    return types.SimpleNamespace(get_torch_profiler=get_torch_profiler)


def test_exported_torch_traces_are_registered_with_their_step_and_role(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path / "out"))
    module = _module()
    assert torch_traces.patch_torch_profile(module) is True
    assert torch_traces.patch_torch_profile(module) is False

    profiler = module.get_torch_profiler([], str(tmp_path), role="train", save_file_prefix="actor", profile_step=3)
    trace = tmp_path / "actor_train_step3_rank0_pid1_20260930151150960.json.gz"
    profiler.export_chrome_trace(str(trace))

    assert profiler.exported == [str(trace)]
    record = json.loads(record_path(tmp_path / "out").read_text())
    assert record["artifacts"] == [
        {"kind": "torch", "file": "../" + trace.name, "global_step": 3, "role": "actor_train"}
    ]
