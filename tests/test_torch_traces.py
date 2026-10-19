import json
import threading
import types

from rl_trace_observer.integrations.verl import torch_traces
from rl_trace_observer.process_record import record_path


class _Profiler:
    def __init__(self):
        self.exported = []

    def export_chrome_trace(self, path):
        self.exported.append(path)

    def stop(self):
        pass


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
    torch_traces.wait_for_exports()

    assert profiler.exported == [str(trace)]
    record = json.loads(record_path(tmp_path / "out").read_text())
    assert record["artifacts"] == [
        {"kind": "torch", "file": "../" + trace.name, "global_step": 3, "role": "actor_train"}
    ]


def test_traces_are_exported_in_the_background(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path / "out"))
    release = threading.Event()

    class SlowProfiler(_Profiler):
        def export_chrome_trace(self, path):
            release.wait(5)
            super().export_chrome_trace(path)

    module = types.SimpleNamespace(get_torch_profiler=lambda *args, **kwargs: SlowProfiler())
    torch_traces.patch_torch_profile(module)
    profiler = module.get_torch_profiler([], str(tmp_path), profile_step=1)

    profiler.export_chrome_trace(str(tmp_path / "trace.json.gz"))  # returns before writing
    assert profiler.exported == []
    release.set()
    torch_traces.wait_for_exports()
    assert profiler.exported == [str(tmp_path / "trace.json.gz")]


def test_finish_hook_waits_for_exports(monkeypatch):
    calls = []
    monkeypatch.setattr(torch_traces, "wait_for_exports", lambda: calls.append("wait"))

    class DistProfiler:
        def __init__(self, command):
            self._finish_hook_cmd, self._relocate_results = command, False

        def _run_finish_hook(self, run_command=True):
            calls.append("hook")

    module = types.SimpleNamespace(DistProfiler=DistProfiler)
    assert torch_traces.patch_finish_hook(module) is True
    DistProfiler(None)._run_finish_hook()
    DistProfiler("upload $SAVE_PATH")._run_finish_hook()
    assert calls == ["hook", "wait", "hook"]


def test_a_scheduled_profiler_exports_each_window_even_when_the_thread_starts_late(tmp_path, monkeypatch):
    import gzip
    import time

    import torch

    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path / "out"))
    # The thread runs after the next window has begun, as under load.
    export = torch_traces._export
    monkeypatch.setattr(torch_traces, "_export", lambda *args: (time.sleep(0.5), export(*args)))
    traces = tmp_path / "traces"
    traces.mkdir()

    windows = iter(range(2))

    def on_trace_ready(profiler):
        profiler.export_chrome_trace(str(traces / f"window{next(windows)}.json.gz"))

    module = types.SimpleNamespace(
        get_torch_profiler=lambda **kwargs: torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=2),
            on_trace_ready=on_trace_ready,
        )
    )
    torch_traces.patch_torch_profile(module)
    profiler = module.get_torch_profiler(profile_step=1)
    profiler.start()
    for window in range(2):
        with torch.profiler.record_function(f"window{window}_marker"):
            torch.ones(8).sum()
        profiler.step()
    profiler.stop()
    torch_traces.wait_for_exports()

    for window in range(2):
        names = {event.get("name") for event in json.load(gzip.open(traces / f"window{window}.json.gz"))["traceEvents"]}
        assert f"window{window}_marker" in names and f"window{1 - window}_marker" not in names


def _recorded_spans(monkeypatch):
    import sys

    recorded = []

    class Logger:
        @staticmethod
        def enabled():
            return True

        @staticmethod
        def trace_span(name, **kwargs):
            recorded.append((name, kwargs))

    monkeypatch.setitem(sys.modules, "verl.utils.tracking", types.SimpleNamespace(RLInsightLogger=Logger))
    return recorded


def test_each_stop_is_a_span_with_its_parts(tmp_path, monkeypatch):
    import torch

    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path / "out"))
    spans = _recorded_spans(monkeypatch)
    module = types.SimpleNamespace(
        get_torch_profiler=lambda contents, save_path, rank=0, profile_step=None: torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=lambda profiler: profiler.export_chrome_trace(str(tmp_path / "t.json.gz")),
        )
    )
    torch_traces.patch_torch_profile(module)
    profiler = module.get_torch_profiler([], str(tmp_path), rank=2, profile_step=3)
    profiler.start()
    torch.ones(8).sum()
    profiler.stop()
    torch_traces.wait_for_exports()

    [(name, span)] = spans
    attributes = span["attributes"]
    assert name == "torch_profiler_stop" and attributes["state_lane_id"] == "rank_2/profiler"
    assert (attributes["backend"], attributes["profile_step"]) == ("kineto", 3)
    # Kineto's disable call, handing off the export, and the rest add up to the stop.
    parts = attributes["disable_ms"] + attributes["export_dispatch_ms"] + attributes["other_ms"]
    assert abs(parts - attributes["total_ms"]) < 0.01
    assert attributes["export_dispatch_ms"] > 0
    assert (tmp_path / "t.json.gz").exists()


def test_the_cupti_monitor_backend_is_requested_from_torch(tmp_path, monkeypatch):
    import torch

    monkeypatch.setenv(torch_traces.PROFILER_BACKEND_ENV, "cupti_monitor")
    module = types.SimpleNamespace(
        # VERL's shape: torch.profiler.profile(**kwargs), with no experimental_config.
        get_torch_profiler=lambda contents, save_path, profile_step=None: torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        )
    )
    torch_traces.patch_torch_profile(module)

    profiler = module.get_torch_profiler([], str(tmp_path), profile_step=1)

    assert profiler._use_cupti_monitor and profiler._cupti_async_export
    assert torch.profiler.profile is torch.profiler.profiler.profile, "torch is left as it was"
    monkeypatch.delenv(torch_traces.PROFILER_BACKEND_ENV)
    assert not module.get_torch_profiler([], str(tmp_path))._use_cupti_monitor


def test_a_cupti_monitor_export_is_finished_off_the_training_thread(tmp_path, monkeypatch):
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path / "out"))
    joined = []

    class Observer:
        def join(self, *, force=True):
            joined.append((threading.current_thread().name, force))
            (tmp_path / "trace.json.gz").write_text("{}")

    class CuptiProfiler(_Profiler):
        _use_cupti_monitor = True

        def take_pending_cupti_export(self):
            return Observer()

    module = types.SimpleNamespace(get_torch_profiler=lambda contents, save_path, profile_step=None: CuptiProfiler())
    torch_traces.patch_torch_profile(module)
    profiler = module.get_torch_profiler([], str(tmp_path), profile_step=4)

    profiler.export_chrome_trace(str(tmp_path / "trace.json.gz"))
    torch_traces.wait_for_exports()

    assert profiler.exported == [str(tmp_path / "trace.json.gz")]
    assert joined == [("rl-trace-torch-export", False)]
    record = json.loads(record_path(tmp_path / "out").read_text())
    assert [(a["file"], a["global_step"]) for a in record["artifacts"]] == [("../trace.json.gz", 4)]
