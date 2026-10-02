"""Export VERL's Torch traces in the background and register them in this process's record.

VERL builds every Torch profiler with ``get_torch_profiler`` and writes each
trace with ``export_chrome_trace`` while it stops the profiler, so the worker,
and the driver waiting for it, are held up for as long as serializing and
compressing the trace takes (tens of seconds for a profiled GPU step). The
wrapper writes the trace on a thread of its own instead, then records its path,
step and role, so the merger never has to infer from a filename, pid or rank
which process wrote it.

Whatever reads the trace must wait for it with :func:`wait_for_exports`: VERL's
finish hook (relocation or the user's command) does so here, and the trainer
does so for every worker before training ends (see ``steps``).
"""

import functools
import inspect
import logging
import threading
from pathlib import Path
from types import ModuleType

from rl_trace_observer.integrations.verl.import_hook import when_imported

logger = logging.getLogger(__name__)

TORCH_PROFILE_MODULE = "verl.utils.profiler.torch_profile"
PROFILE_MODULE = "verl.utils.profiler.profile"
_PATCHED = "_rl_trace_observer_registers_traces"

_exports: list[threading.Thread] = []
_exports_lock = threading.Lock()


def _register(path: str, global_step: int | None, role: str | None) -> None:
    try:
        from rl_trace_observer.output import trace_output_dir
        from rl_trace_observer.process_record import register_artifact

        register_artifact(trace_output_dir(), "torch", Path(path), global_step=global_step, role=role)
    except Exception:
        logger.exception("Failed to register the Torch trace %s", path)


def _export(export, path: str, global_step: int | None, role: str | None) -> None:
    try:
        export(path)
    except Exception:
        logger.exception("Failed to export the Torch trace %s", path)
        return
    _register(path, global_step, role)


def wait_for_exports() -> None:
    """Block until every trace this process started exporting is written and registered."""
    while True:
        with _exports_lock:
            pending = [thread for thread in _exports if thread.is_alive()]
            _exports[:] = pending
        if not pending:
            return
        for thread in pending:
            thread.join()


def patch_torch_profile(module: ModuleType) -> bool:
    """Wrap ``module.get_torch_profiler``; returns ``False`` if already wrapped."""
    original = module.get_torch_profiler
    if getattr(original, _PATCHED, False):
        return False
    signature = inspect.signature(original)

    @functools.wraps(original)
    def get_torch_profiler(*args, **kwargs):
        profiler = original(*args, **kwargs)
        arguments = signature.bind(*args, **kwargs).arguments
        step = arguments.get("profile_step")
        step = step if isinstance(step, int) else None
        # The same parts VERL puts in front of the step in the filename, e.g. "actor_train".
        role = "_".join(str(part) for part in (arguments.get("save_file_prefix"), arguments.get("role")) if part)
        export = profiler.export_chrome_trace

        def export_chrome_trace(path):
            # Not a daemon: a normal interpreter exit still finishes the file.
            thread = threading.Thread(
                target=_export, args=(export, path, step, role or None), name="rl-trace-torch-export"
            )
            with _exports_lock:
                _exports.append(thread)
            thread.start()

        profiler.export_chrome_trace = export_chrome_trace
        return profiler

    setattr(get_torch_profiler, _PATCHED, True)
    module.get_torch_profiler = get_torch_profiler
    return True


def patch_finish_hook(module: ModuleType) -> bool:
    """Make ``DistProfiler``'s finish hook (relocation, the user's command) see finished traces."""
    profiler_class = module.DistProfiler
    original = profiler_class._run_finish_hook
    if getattr(original, _PATCHED, False):
        return False

    @functools.wraps(original)
    def _run_finish_hook(self, *args, **kwargs):
        if getattr(self, "_relocate_results", False) or getattr(self, "_finish_hook_cmd", None):
            wait_for_exports()
        return original(self, *args, **kwargs)

    setattr(_run_finish_hook, _PATCHED, True)
    profiler_class._run_finish_hook = _run_finish_hook
    return True


def install() -> None:
    when_imported(TORCH_PROFILE_MODULE, patch_torch_profile)
    when_imported(PROFILE_MODULE, patch_finish_hook)
