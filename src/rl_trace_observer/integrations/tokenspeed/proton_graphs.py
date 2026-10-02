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
  and moves the file to the name ``/stop_profile`` would have written. The
  writing runs in the background (see ``profile_saving``, which routes
  TokenSpeed's Proton calls here).

Proton completes a phase only when a later phase receives a kernel, so each
completion launches one tiny kernel. TokenSpeed itself is not modified.
"""

import functools
import os
import shutil
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

    def stop(self) -> int:
        """End the profiled phase; returns it for :meth:`write`.

        Kernels keep landing in the next phase, which is discarded, until
        :meth:`write` deactivates the session.
        """
        phase, self.phase = self.phase, None
        self.proton.data.advance_phase(self.id)
        return phase

    def write(self, phase: int, output: str) -> Path:
        """Write a stopped ``phase`` to ``<output>.chrome_trace``; may run on another thread.

        The next :meth:`begin` must wait for it.
        """
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


def session() -> Session | None:
    """This scheduler's long-lived session, once it is built; ``profile_saving`` drives it."""
    return _session


def install() -> None:
    if not os.environ.get(ENV):
        return
    from rl_trace_observer.integrations.verl.import_hook import when_imported

    when_imported("tokenspeed.runtime.engine.event_loop", _patch_event_loop)
