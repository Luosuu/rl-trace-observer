"""Proton profiles of a TokenSpeed server that replays CUDA graphs.

Proton attributes a replayed kernel to its graph node only when the session was
active while the graph was captured; otherwise the kernel hangs off the graph
launch, and finalizing a trace-mode session fails with "Cannot find CPU scope
event for kernel launch". ``tokenspeed serve`` captures its graphs at startup
and starts a new Proton session on every ``/start_profile``, so its Proton
profiles need ``--enforce-eager``.

With :data:`ENV` set to a directory, :func:`install` (run in every process of
``tokenspeed serve`` through ``sitecustomize``, see ``weight_group.SITE_DIR``)
keeps one Proton session per scheduler for the life of the process instead:

- it starts before the scheduler is built, so it sees every capture, and is
  deactivated once the scheduler is built;
- ``/start_profile`` moves it to a new data phase and activates it;
- ``/stop_profile`` completes that phase, which makes Proton's periodic flushing
  write it as ``<session>.part_<phase>.chrome_trace``, deactivates the session,
  and moves the file to the name ``/stop_profile`` would have written.

Proton completes a phase only when a later phase receives a kernel, so each
completion launches one tiny kernel. TokenSpeed itself is not modified.
"""

import functools
import os
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType

ENV = "RL_TRACE_TOKENSPEED_PROTON_SESSION_DIR"
FLUSH_TIMEOUT_ENV = "RL_TRACE_TOKENSPEED_PROTON_FLUSH_TIMEOUT"
FORMAT = "chrome_trace"
_PATCHED = "_rl_trace_observer_proton_graphs"


class Session:
    """A scheduler's long-lived Proton session."""

    def __init__(self, proton, path: Path):
        self.proton = proton
        self.path = path
        self.id: int | None = None
        self.device: int | None = None
        self.phase: int | None = None  # the phase being profiled
        self._stream = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.id = self.proton.start(
            str(self.path), data="trace", hook="triton", mode=f"periodic_flushing:format={FORMAT}"
        )

    def park(self, device: int) -> None:
        """Stop recording after startup; what capture recorded stays with the session."""
        self.device = device
        self._complete(self.proton.data.advance_phase(self.id) - 1).unlink()

    def begin(self) -> None:
        self.phase = self.proton.data.advance_phase(self.id)
        self.proton.activate(self.id)

    def end(self, output: str) -> Path:
        """Write the profiled phase to ``<output>.chrome_trace``."""
        phase, self.phase = self.phase, None
        self.proton.data.advance_phase(self.id)
        part = self._complete(phase)
        target = Path(f"{output}.{FORMAT}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(part, target)
        for stale in self.path.parent.glob(f"{self.path.name}.part_*.{FORMAT}"):
            stale.unlink(missing_ok=True)  # the phases between profiles
        return target

    def _complete(self, phase: int) -> Path:
        """Flush ``phase`` (older than the current one) to its file and deactivate."""
        self._deactivate_after_kernel()
        part = self.path.with_name(f"{self.path.name}.part_{phase}.{FORMAT}")
        _wait_written(part, lambda: self.proton.data.is_phase_complete(self.id, phase))
        return part

    def _deactivate_after_kernel(self) -> None:
        import torch

        # Runs on TokenSpeed's control-plane thread too, whose dispatch guard
        # rejects tensor ops; `_sleep` launches its kernel without dispatch.
        with torch.cuda.device(self.device):
            if self._stream is None:
                self._stream = torch.cuda.Stream()
            with torch.cuda.stream(self._stream):
                torch.cuda._sleep(1)
            self.proton.deactivate(self.id, flushing=True)


def _wait_written(path: Path, complete) -> None:
    """Wait until Proton has written ``path``.

    Proton marks a phase complete before it writes the phase, so the file must
    also hold a whole JSON object whose size has stopped changing.
    """
    deadline = time.monotonic() + float(os.environ.get(FLUSH_TIMEOUT_ENV, "300"))
    size = -1
    while time.monotonic() < deadline:
        if complete() and path.is_file():
            current = path.stat().st_size
            if current == size and current > 0 and _ends_json_object(path):
                return
            size = current
        time.sleep(0.2)
    raise TimeoutError(f"Proton did not write {path}")


def _ends_json_object(path: Path) -> bool:
    with path.open("rb") as file:
        file.seek(max(0, path.stat().st_size - 16))
        return file.read().rstrip().endswith(b"}")


_session: Session | None = None


def _current_device() -> int:
    import torch

    return torch.cuda.current_device()


def _patch_event_loop(module: ModuleType) -> None:
    original = module.EventLoop.__init__
    if getattr(original, _PATCHED, False):
        return

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        global _session
        from tokenspeed_kernel.profiling import proton, proton_available

        if _session is not None or not proton_available():
            return original(self, *args, **kwargs)
        session = Session(proton, Path(os.environ[ENV]) / f"proton-{os.getpid()}")
        session.start()
        try:
            original(self, *args, **kwargs)
        except BaseException:
            proton.finalize(session.id)
            raise
        session.park(_current_device())
        _session = session

    setattr(init, _PATCHED, True)
    module.EventLoop.__init__ = init
    # event_loop imports request_handler, which calls TokenSpeed's Proton helpers.
    _patch_request_handler(sys.modules["tokenspeed.runtime.engine.request_handler"])


def _patch_request_handler(module: ModuleType) -> None:
    """Route TokenSpeed's Proton start/stop to the long-lived session."""
    if getattr(module.start_profiling, _PATCHED, False):
        return
    original_start, original_stop = module.start_profiling, module.stop_profiling

    @functools.wraps(original_start)
    def start_profiling(config=None):
        from tokenspeed_kernel.profiling import ProfilingState

        state = ProfilingState.get()
        if _session is None or config is None or state.active:
            return original_start(config)
        _session.begin()
        # kernel_scope() records CPU scopes only while this state is active.
        state._config, state._session, state.enabled = config, _session.id, True
        return _session.id

    @functools.wraps(original_stop)
    def stop_profiling():
        from tokenspeed_kernel.profiling import ProfilingState

        state = ProfilingState.get()
        if _session is None or _session.phase is None or state._session != _session.id:
            return original_stop()
        output = state._config.output
        state._config, state._session, state.enabled = None, None, False
        _session.end(output)

    for function in (start_profiling, stop_profiling):
        setattr(function, _PATCHED, True)
    module.start_profiling, module.stop_profiling = start_profiling, stop_profiling


def install() -> None:
    if not os.environ.get(ENV):
        return
    from rl_trace_observer.integrations.verl.import_hook import when_imported

    when_imported("tokenspeed.runtime.engine.event_loop", _patch_event_loop)
