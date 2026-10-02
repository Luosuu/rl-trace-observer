"""Write TokenSpeed's VizTracer and Proton profiles in the background.

``/stop_profile`` makes each scheduler write its VizTracer report and Proton
trace before it replies, and its TP ranks wait for each other: for a profiled
GPU step that holds up the scheduler, and every caller waiting for the reply,
for seconds. Installed in every process of ``tokenspeed serve`` through
``sitecustomize`` (see ``weight_group.SITE_DIR``), this patch only stops
recording there and writes the files on threads of their own:

- VizTracer: TokenSpeed's ``VizTracer`` is replaced by a subclass whose
  ``save()`` runs on a thread;
- Proton: ``stop_profiling`` deactivates the session and finalizes it on a
  thread; with the long-lived session of ``proton_graphs`` (CUDA graphs), it
  ends the profiled phase and writes it on a thread.

The next ``/start_profile`` waits for the previous profile's files first: a new
VizTracer reuses the process's tracer, and a Proton session is restarted or
reactivated. Files appear after ``/stop_profile`` returns; the server actor
registers each one once it is complete (see ``server``). TokenSpeed itself is
not modified.
"""

import functools
import logging
import threading
from types import ModuleType

logger = logging.getLogger(__name__)

REQUEST_HANDLER_MODULE = "tokenspeed.runtime.engine.request_handler"
_PATCHED = "_rl_trace_observer_background_saves"

_saves: list[threading.Thread] = []
_saves_lock = threading.Lock()


def in_background(function, *args, name: str) -> None:
    """Run ``function(*args)`` on a thread; :func:`wait` waits for it."""

    def run():
        try:
            function(*args)
        except Exception:
            logger.exception("rl-trace-observer: %s failed", name)

    # Not a daemon: a normal interpreter exit still finishes the file.
    thread = threading.Thread(target=run, name=f"rl-trace-{name}")
    with _saves_lock:
        _saves.append(thread)
    thread.start()


def wait() -> None:
    """Block until every profile this process started writing is written."""
    while True:
        with _saves_lock:
            pending = [thread for thread in _saves if thread.is_alive()]
            _saves[:] = pending
        if not pending:
            return
        for thread in pending:
            thread.join()


def _viztracer_class(base: type) -> type:
    class BackgroundSaveVizTracer(base):
        """``save()`` returns at once and writes the report on a thread."""

        def __init__(self, *args, **kwargs):
            # The process has one tracer; the previous report must be out of it.
            wait()
            super().__init__(*args, **kwargs)

        def save(self, *args, **kwargs):
            in_background(functools.partial(base.save, self, *args, **kwargs), name="viztracer-save")

    BackgroundSaveVizTracer.__name__ = BackgroundSaveVizTracer.__qualname__ = base.__name__
    setattr(BackgroundSaveVizTracer, _PATCHED, True)
    return BackgroundSaveVizTracer


def _patch(module: ModuleType) -> None:
    if getattr(module.start_profiling, _PATCHED, False):
        return
    from . import proton_graphs

    original_start = module.start_profiling

    @functools.wraps(original_start)
    def start_profiling(config=None):
        from tokenspeed_kernel.profiling import ProfilingState

        wait()
        state = ProfilingState.get()
        session = proton_graphs.session()
        if session is None or config is None or state.active:
            return original_start(config)
        session.begin()
        # kernel_scope() records CPU scopes only while this state is active.
        state._config, state._session, state.enabled = config, session.id, True
        return session.id

    @functools.wraps(module.stop_profiling)
    def stop_profiling():
        from tokenspeed_kernel.profiling import ProfilingState, proton

        state = ProfilingState.get()
        if not state.active:
            state._config, state._session, state.enabled = None, None, False
            return
        session_id, config = state._session, state._config
        state._config, state._session, state.enabled = None, None, False
        session = proton_graphs.session()
        if session is not None and session.phase is not None and session_id == session.id:
            phase = session.stop()
            in_background(session.write, phase, config.output, name="proton-write")
            return
        # Stop recording now; flushing and writing happen in finalize.
        proton.deactivate(session_id)
        in_background(proton.finalize, session_id, config.output_format if config else "", name="proton-finalize")

    for function in (start_profiling, stop_profiling):
        setattr(function, _PATCHED, True)
    module.start_profiling, module.stop_profiling = start_profiling, stop_profiling
    module.VizTracer = _viztracer_class(module.VizTracer)


def install() -> None:
    from rl_trace_observer.integrations.verl.import_hook import when_imported

    when_imported(REQUEST_HANDLER_MODULE, _patch)
