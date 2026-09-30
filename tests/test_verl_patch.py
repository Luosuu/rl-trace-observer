import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rl_trace_observer.integrations.verl.patch import install, is_installed, uninstall
from rl_trace_observer.observer import ObserverRegistry


class FakeDistProfiler:
    def __init__(self, *, enabled=True, selected=True, rank=0):
        self.rank = rank
        self.config = SimpleNamespace(enable=enabled)
        self.tool_config = SimpleNamespace()
        self.save_file_prefix = "actor"
        self._tool = "torch"
        self._enabled = enabled
        self._selected = selected
        self.calls = []

    def check_enable(self):
        return self._enabled

    def check_this_rank(self):
        return self._selected

    def start(self, **kwargs):
        self.calls.append(("backend_start", kwargs))
        return "start-result"

    def stop(self, run_command=True):
        self.calls.append(("backend_stop", run_command))
        return "stop-result"


@pytest.fixture(autouse=True)
def clean_patch_and_registry():
    if is_installed():
        uninstall()
    ObserverRegistry._factories.clear()
    yield
    if is_installed():
        uninstall()
    ObserverRegistry._factories.clear()


def test_stop_forwards_verl_arguments():
    ObserverRegistry.register("test", Mock)
    install(FakeDistProfiler)
    profiler = FakeDistProfiler()

    profiler.start()
    profiler.stop(run_command=False)

    assert profiler.calls[-1] == ("backend_stop", False)


def test_observer_wraps_backend_and_receives_verl_context():
    calls = []

    class Observer:
        def on_start(self, context):
            calls.append(("observer_start", context))

        def on_stop(self, context):
            calls.append(("observer_stop", context))

    ObserverRegistry.register("test", Observer)
    assert install(FakeDistProfiler)
    profiler = FakeDistProfiler(rank=3)

    assert profiler.start(profile_step=42, role="e2e") == "start-result"
    calls.append(profiler.calls[-1])
    assert profiler.stop() == "stop-result"
    calls.insert(-1, profiler.calls[-1])

    assert [name for name, _ in calls] == [
        "observer_start",
        "backend_start",
        "backend_stop",
        "observer_stop",
    ]
    context = calls[0][1]
    assert context is calls[-1][1]
    assert context.rank == 3
    assert context.tool == "torch"
    assert context.save_file_prefix == "actor"
    assert context.metadata == {"profile_step": 42, "role": "e2e"}


def test_disabled_or_unselected_profiler_bypasses_observers():
    observer = Mock()
    ObserverRegistry.register("test", lambda: observer)
    install(FakeDistProfiler)

    for profiler in (FakeDistProfiler(enabled=False), FakeDistProfiler(selected=False)):
        profiler.start(profile_step=1)
        profiler.stop()

    observer.on_start.assert_not_called()
    observer.on_stop.assert_not_called()


def test_observer_start_failure_does_not_interrupt_backend():
    observer = Mock()
    observer.on_start.side_effect = RuntimeError("observer failed")
    ObserverRegistry.register("test", lambda: observer)
    install(FakeDistProfiler)
    profiler = FakeDistProfiler()

    assert profiler.start(profile_step=1) == "start-result"
    assert profiler.stop() == "stop-result"

    observer.on_stop.assert_not_called()
    assert [name for name, _ in profiler.calls] == ["backend_start", "backend_stop"]


def test_backend_start_failure_stops_active_observers():
    observer = Mock()
    ObserverRegistry.register("test", lambda: observer)

    class FailingProfiler(FakeDistProfiler):
        def start(self, **kwargs):
            raise RuntimeError("backend failed")

    install(FailingProfiler)
    profiler = FailingProfiler()

    with pytest.raises(RuntimeError, match="backend failed"):
        profiler.start(profile_step=7)

    observer.on_start.assert_called_once()
    observer.on_stop.assert_called_once_with(observer.on_start.call_args.args[0])


def test_install_is_idempotent_and_uninstall_restores_methods():
    original_start = FakeDistProfiler.start
    original_stop = FakeDistProfiler.stop

    assert install(FakeDistProfiler)
    assert not install(FakeDistProfiler)
    assert FakeDistProfiler.start is not original_start
    assert FakeDistProfiler.stop is not original_stop

    assert uninstall()
    assert not uninstall()
    assert FakeDistProfiler.start is original_start
    assert FakeDistProfiler.stop is original_stop


def test_viztracer_observer_writes_actor_trace(tmp_path, monkeypatch):
    pytest.importorskip("viztracer")
    from rl_trace_observer.integrations.verl.viztracer import VizTracerObserver

    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("RL_TRACE_VIZTRACER_MIN_DURATION_US", "0")
    ObserverRegistry.register("viztracer", VizTracerObserver)
    install(FakeDistProfiler)
    profiler = FakeDistProfiler(rank=2)

    profiler.start(profile_step=9, role="actor")
    sum(range(10))
    profiler.stop()

    traces = list(tmp_path.glob("step-9-role-actor-rank-2-pid-*.viztracer.json"))
    assert len(traces) == 1
    assert json.loads(traces[0].read_text())["traceEvents"]
