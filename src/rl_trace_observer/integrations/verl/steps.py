"""Mark each VERL training step on the trainer's RL-Insight timeline.

RL-Insight spans carry no step, so the driver records one ``global_step`` span
per step of VERL's v1 trainer (``PPOTrainer.step``: rollout and training). The
merger uses these spans as the time window of each profile session.

``PPOTrainer`` lives in a heavy module that the plugin must not import itself,
so the patch is applied when VERL imports it.
"""

import functools
import importlib.abc
import importlib.machinery
import logging
import sys
import time
from collections.abc import Callable
from types import ModuleType

from rl_trace_observer.context import STEP_SPAN_NAME, current_run_id

logger = logging.getLogger(__name__)

TRAINER_MODULE = "verl.trainer.ppo.v1.trainer_base"
_PATCHED = "_rl_trace_observer_step_marker"


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, callback: Callable[[ModuleType], None]):
        self._loader = loader
        self._callback = callback

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        self._callback(module)

    def __getattr__(self, name: str):
        # get_source, get_filename, ...: tracebacks and inspect keep working.
        return getattr(self._loader, name)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Run a callback right after one module is imported."""

    def __init__(self, name: str, callback: Callable[[ModuleType], None]):
        self.name = name
        self._callback = callback

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _PostImportLoader(spec.loader, self._callback)
                return spec
        return None


def when_imported(name: str, callback: Callable[[ModuleType], None]) -> None:
    """Call ``callback(module)`` now if ``name`` is imported, else right after it is."""
    if name in sys.modules:
        callback(sys.modules[name])
        return
    if not any(isinstance(finder, _PostImportFinder) and finder.name == name for finder in sys.meta_path):
        sys.meta_path.insert(0, _PostImportFinder(name, callback))


def _emit_step_span(trainer: object, start_time_ns: int, end_time_ns: int) -> None:
    from verl.utils.tracking import RLInsightLogger

    if not RLInsightLogger.enabled():
        return
    global_step = getattr(trainer, "global_steps", None)
    if global_step is None:
        return
    attributes = {"state_lane_id": "trainer", "global_step": int(global_step)}
    if run_id := current_run_id():
        attributes["run_id"] = run_id
    RLInsightLogger.trace_span(
        STEP_SPAN_NAME, start_time_ns=start_time_ns, end_time_ns=end_time_ns, attributes=attributes
    )
    try:
        from rl_trace_observer.output import trace_output_dir
        from rl_trace_observer.process_record import register_artifact

        register_artifact(trace_output_dir(), None, None, role="trainer")
    except Exception:
        logger.exception("Failed to record the trainer role")


def patch_trainer(module: ModuleType) -> bool:
    """Wrap ``module.PPOTrainer.step``; returns ``False`` if already wrapped."""
    trainer_class = module.PPOTrainer
    original = trainer_class.step
    if getattr(original, _PATCHED, False):
        return False

    @functools.wraps(original)
    def step(self, *args, **kwargs):
        start_time_ns = time.time_ns()
        try:
            return original(self, *args, **kwargs)
        finally:
            try:
                _emit_step_span(self, start_time_ns, time.time_ns())
            except Exception:
                logger.exception("Failed to record the global_step span")

    setattr(step, _PATCHED, True)
    trainer_class.step = step
    return True


def install() -> None:
    when_imported(TRAINER_MODULE, patch_trainer)
