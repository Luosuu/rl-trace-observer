"""Register the Torch traces VERL exports in this process's record.

VERL builds every Torch profiler with ``get_torch_profiler`` and writes each
trace with ``export_chrome_trace``. Wrapping the two records the trace's path,
step and role at the moment it is written, so the merger never has to infer
from a filename, pid or rank which process wrote it.
"""

import functools
import inspect
import logging
from pathlib import Path
from types import ModuleType

from rl_trace_observer.integrations.verl.import_hook import when_imported

logger = logging.getLogger(__name__)

TORCH_PROFILE_MODULE = "verl.utils.profiler.torch_profile"
_PATCHED = "_rl_trace_observer_registers_traces"


def _register(path: str, global_step: int | None, role: str | None) -> None:
    try:
        from rl_trace_observer.output import trace_output_dir
        from rl_trace_observer.process_record import register_artifact

        register_artifact(trace_output_dir(), "torch", Path(path), global_step=global_step, role=role)
    except Exception:
        logger.exception("Failed to register the Torch trace %s", path)


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
            result = export(path)
            _register(path, step, role or None)
            return result

        profiler.export_chrome_trace = export_chrome_trace
        return profiler

    setattr(get_torch_profiler, _PATCHED, True)
    module.get_torch_profiler = get_torch_profiler
    return True


def install() -> None:
    when_imported(TORCH_PROFILE_MODULE, patch_torch_profile)
