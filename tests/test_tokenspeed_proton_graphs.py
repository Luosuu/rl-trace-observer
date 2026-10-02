"""One Proton session per TokenSpeed scheduler, written out one phase per profile."""

import json
import sys
import types
from pathlib import Path

import pytest

from rl_trace_observer.integrations.tokenspeed import proton_graphs


class FakeProton:
    """Proton's session, phase and periodic-flushing behaviour, without a GPU."""

    def __init__(self):
        self.path = None
        self.phase = 0
        self.active = False
        self.written = -1
        self.records = {}  # phase -> names recorded in it
        self.data = types.SimpleNamespace(
            advance_phase=self.advance_phase, is_phase_complete=lambda session, phase: phase <= self.written
        )

    def start(self, path, *, data, hook, mode):
        assert (data, hook, mode) == ("trace", "triton", "periodic_flushing:format=chrome_trace")
        self.path, self.active = path, True
        return 7

    def record(self, name):
        if self.active:
            self.records.setdefault(self.phase, []).append(name)

    def advance_phase(self, session):
        assert session == 7
        self.phase += 1
        return self.phase

    def activate(self, session):
        self.active = True

    def deactivate(self, session, flushing=False):
        # Periodic flushing writes the phases before the latest one with a kernel.
        if flushing and self.phase in self.records:
            for phase in range(self.written + 1, self.phase):
                names = self.records.pop(phase, [])
                Path(f"{self.path}.part_{phase}.chrome_trace").write_text(json.dumps({"traceEvents": names}))
            self.written = self.phase - 1
        self.active = False

    def finalize(self, session):
        self.active = False


class ProfilingState:
    """tokenspeed_kernel.profiling.ProfilingState: what kernel_scope() checks."""

    def __init__(self):
        self.enabled, self._session, self._config = False, None, None

    @property
    def active(self):
        return self.enabled and self._session is not None


@pytest.fixture
def tokenspeed(monkeypatch, tmp_path):
    proton = FakeProton()
    state = ProfilingState()
    state_cls = types.SimpleNamespace(get=lambda: state)
    finalized = []

    def original_start(config=None):
        raise AssertionError("TokenSpeed must not start its own Proton session")

    profiling = types.ModuleType("tokenspeed_kernel.profiling")
    profiling.proton, profiling.proton_available, profiling.ProfilingState = proton, lambda: True, state_cls
    monkeypatch.setitem(sys.modules, "tokenspeed_kernel.profiling", profiling)

    handler = types.ModuleType("tokenspeed.runtime.engine.request_handler")
    handler.start_profiling, handler.stop_profiling = original_start, lambda: finalized.append(True)
    monkeypatch.setitem(sys.modules, handler.__name__, handler)

    class EventLoop:
        def __init__(self):
            proton.record("graph capture")

    event_loop = types.ModuleType("tokenspeed.runtime.engine.event_loop")
    event_loop.EventLoop = EventLoop
    monkeypatch.setitem(sys.modules, event_loop.__name__, event_loop)

    monkeypatch.setenv(proton_graphs.ENV, str(tmp_path / "session"))
    monkeypatch.setattr(proton_graphs, "_session", None)
    monkeypatch.setattr(proton_graphs, "_current_device", lambda: 0)
    monkeypatch.setattr(
        proton_graphs.Session,
        "_deactivate_after_kernel",
        lambda self: (proton.record("marker"), proton.deactivate(self.id, flushing=True)),
    )
    proton_graphs.install()
    yield types.SimpleNamespace(
        proton=proton, state=state, handler=handler, event_loop=event_loop, finalized=finalized, tmp=tmp_path
    )


def test_session_spans_capture_and_writes_each_profile(tokenspeed):
    t = tokenspeed
    t.event_loop.EventLoop()
    assert t.proton.path == str(t.tmp / "session" / f"proton-{__import__('os').getpid()}")
    assert not t.proton.active, "the session is parked after startup"
    assert not list((t.tmp / "session").iterdir()), "the startup phase is not kept"

    for step in (3, 4):
        output = t.tmp / f"step-{step}" / f"run-step-{step}-TP0.proton"
        config = types.SimpleNamespace(output=str(output))
        assert t.handler.start_profiling(config) == 7
        assert t.state.enabled and t.state._session == 7 and t.proton.active
        t.proton.record(f"decode step {step}")
        t.handler.stop_profiling()
        assert not t.state.enabled and not t.proton.active
        written = Path(f"{output}.chrome_trace")
        assert json.loads(written.read_text()) == {"traceEvents": [f"decode step {step}"]}
        assert not list((t.tmp / "session").iterdir()), "phases between profiles are removed"
    assert t.finalized == []


def test_without_a_session_tokenspeed_keeps_its_own(tokenspeed, monkeypatch):
    t = tokenspeed
    t.event_loop.EventLoop()
    monkeypatch.setattr(proton_graphs, "_session", None)
    with pytest.raises(AssertionError, match="own Proton session"):
        t.handler.start_profiling(types.SimpleNamespace(output="x"))
    t.handler.stop_profiling()
    assert t.finalized == [True]


def test_disabled_without_the_environment(monkeypatch):
    monkeypatch.delenv(proton_graphs.ENV, raising=False)
    event_loop = types.ModuleType("tokenspeed.runtime.engine.event_loop")
    event_loop.EventLoop = type("EventLoop", (), {})
    monkeypatch.setitem(sys.modules, event_loop.__name__, event_loop)
    init = event_loop.EventLoop.__init__
    proton_graphs.install()
    assert event_loop.EventLoop.__init__ is init
