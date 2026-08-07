import logging
from typing import Any

from rl_trace_observer.observer import ObserverContext, ObserverRegistry

logger = logging.getLogger(__name__)

_STATE_ATTRIBUTE = "_rl_trace_observer_lifecycle"
_patched_class: type | None = None
_original_start: Any = None
_original_stop: Any = None
_patched_start: Any = None
_patched_stop: Any = None


def _is_selected(profiler: object) -> bool:
    return bool(profiler.check_enable() and profiler.check_this_rank())


def _stop_observers(profiler: object) -> None:
    lifecycle = getattr(profiler, _STATE_ATTRIBUTE, None)
    if lifecycle is None:
        return

    context, active_observers = lifecycle
    for name, observer in reversed(active_observers):
        try:
            observer.on_stop(context)
        except Exception:
            logger.exception("Profiler observer %r failed during stop", name)
    delattr(profiler, _STATE_ATTRIBUTE)


def install(profiler_class: type | None = None) -> bool:
    """Patch VERL's DistProfiler lifecycle.

    Args:
        profiler_class: Injection point used by tests. When omitted, VERL's
            current ``DistProfiler`` class is imported lazily.

    Returns:
        ``True`` when the patch was installed and ``False`` when it was already
        installed on the same class.
    """
    global _original_start, _original_stop, _patched_class, _patched_start, _patched_stop

    if profiler_class is None:
        from verl.utils.profiler.profile import DistProfiler

        profiler_class = DistProfiler

    if _patched_class is profiler_class:
        return False
    if _patched_class is not None:
        raise RuntimeError("The VERL profiler patch is already installed on another class")

    original_start = profiler_class.start
    original_stop = profiler_class.stop

    def patched_start(self, **kwargs):
        if not _is_selected(self):
            return original_start(self, **kwargs)

        context = ObserverContext.from_profiler(self, kwargs)
        active_observers = []
        for name, observer in ObserverRegistry.create_observers():
            try:
                observer.on_start(context)
                active_observers.append((name, observer))
            except Exception:
                logger.exception("Profiler observer %r failed during start; skipping it", name)
        setattr(self, _STATE_ATTRIBUTE, (context, active_observers))

        try:
            return original_start(self, **kwargs)
        except BaseException:
            _stop_observers(self)
            raise

    def patched_stop(self):
        try:
            return original_stop(self)
        finally:
            _stop_observers(self)

    patched_start.__name__ = original_start.__name__
    patched_start.__doc__ = original_start.__doc__
    patched_stop.__name__ = original_stop.__name__
    patched_stop.__doc__ = original_stop.__doc__

    profiler_class.start = patched_start
    profiler_class.stop = patched_stop
    _patched_class = profiler_class
    _original_start = original_start
    _original_stop = original_stop
    _patched_start = patched_start
    _patched_stop = patched_stop
    return True


def uninstall() -> bool:
    """Restore VERL's original methods without clobbering a later patch."""
    global _original_start, _original_stop, _patched_class, _patched_start, _patched_stop

    if _patched_class is None:
        return False
    if _patched_class.start is not _patched_start or _patched_class.stop is not _patched_stop:
        raise RuntimeError("DistProfiler was patched again after RL Trace Observer; refusing to overwrite it")

    _patched_class.start = _original_start
    _patched_class.stop = _original_stop
    _patched_class = None
    _original_start = None
    _original_stop = None
    _patched_start = None
    _patched_stop = None
    return True


def is_installed() -> bool:
    return _patched_class is not None
