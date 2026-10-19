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

Stopping the profiler still holds the worker up: torch synchronizes the device
and Kineto collects and correlates every GPU activity (seconds for a profiled
GPU step). Each stop is a ``torch_profiler_stop`` RL-Insight span on the
worker's ``rank_<r>/profiler`` lane, with how long Kineto's disable call took
(``disable_ms``), how long handing the export off took (``export_dispatch_ms``)
and the rest (``other_ms``, mostly the device synchronization).

``RL_TRACE_TORCH_PROFILER_BACKEND=cupti_monitor`` builds the profiler with
torch's CUPTI-monitor backend instead (it needs ``cupti-python``): Kineto
records only CPU events, a CUPTI monitor records the GPU's, and stopping only
closes the window, without synchronizing the device; the GPU records are
merged into the trace on the export thread.
"""

import copy
import functools
import inspect
import json
import logging
import os
import threading
import time
from pathlib import Path
from types import ModuleType

from rl_trace_observer.integrations.verl.import_hook import require, when_imported

logger = logging.getLogger(__name__)

TORCH_PROFILE_MODULE = "verl.utils.profiler.torch_profile"
PROFILE_MODULE = "verl.utils.profiler.profile"
_PATCHED = "_rl_trace_observer_registers_traces"
PROFILER_BACKEND_ENV = "RL_TRACE_TORCH_PROFILER_BACKEND"
CUPTI_MONITOR = "cupti_monitor"
STOP_SPAN_NAME = "torch_profiler_stop"
# Only one profiler is built with a patched torch.profiler.profile at a time.
_construct_lock = threading.Lock()

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


def _start_export(target, *args) -> None:
    # Not a daemon: a normal interpreter exit still finishes the file.
    thread = threading.Thread(target=target, args=args, name="rl-trace-torch-export")
    with _exports_lock:
        _exports.append(thread)
    thread.start()


def _finish_cupti_export(observer, path: str, global_step: int | None, role: str | None) -> None:
    try:
        # Off the training thread, so without forcing a CUPTI flush (torch's take_pending_cupti_export).
        observer.join(force=False)
    except Exception:
        logger.exception("Failed to export the Torch trace %s", path)
        return
    # The monitor always writes gzipped traces.
    _register(path if path.endswith(".gz") else f"{path}.gz", global_step, role)


def _backend() -> str:
    return os.environ.get(PROFILER_BACKEND_ENV, "").strip().lower() or "kineto"


def _build(original, args, kwargs):
    """Call VERL's ``get_torch_profiler``, with torch's CUPTI-monitor backend if asked for."""
    if _backend() != CUPTI_MONITOR:
        return original(*args, **kwargs)
    import torch.profiler
    from torch._C._profiler import _ExperimentalConfig

    config = _ExperimentalConfig(
        custom_profiler_config=json.dumps({"backend": CUPTI_MONITOR, "cupti_monitor_async_export": True})
    )
    # VERL builds the profiler with torch.profiler.profile(**kwargs) and no way to pass a config.
    with _construct_lock:
        profile = torch.profiler.profile
        torch.profiler.profile = functools.partial(profile, experimental_config=config)
        try:
            return original(*args, **kwargs)
        finally:
            torch.profiler.profile = profile


def _emit_stop(start_time_ns: int, end_time_ns: int, profiler, rank, step, dispatch_ns: int) -> None:
    from verl.utils.tracking import RLInsightLogger

    if not RLInsightLogger.enabled():
        return
    total_ns = end_time_ns - start_time_ns
    attributes = {
        "state_lane_id": f"rank_{rank}/profiler",
        "backend": CUPTI_MONITOR if getattr(profiler, "_use_cupti_monitor", False) else "kineto",
        "total_ms": round(total_ns / 1e6, 3),
        "export_dispatch_ms": round(dispatch_ns / 1e6, 3),
    }
    stats = getattr(getattr(profiler, "profiler", None), "_stats", None)
    disable_us = getattr(stats, "profiler_disable_call_duration_us", None)
    if isinstance(disable_us, int):
        attributes["disable_ms"] = round(disable_us / 1e3, 3)
        attributes["other_ms"] = round((total_ns - dispatch_ns) / 1e6 - disable_us / 1e3, 3)
    if step is not None:
        attributes["profile_step"] = step
    RLInsightLogger.trace_span(
        STOP_SPAN_NAME, start_time_ns=start_time_ns, end_time_ns=end_time_ns, attributes=attributes
    )


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
    require(module, "get_torch_profiler")
    original = module.get_torch_profiler
    if getattr(original, _PATCHED, False):
        return False
    signature = inspect.signature(original)

    @functools.wraps(original)
    def get_torch_profiler(*args, **kwargs):
        profiler = _build(original, args, kwargs)
        arguments = signature.bind(*args, **kwargs).arguments
        step = arguments.get("profile_step")
        step = step if isinstance(step, int) else None
        # The same parts VERL puts in front of the step in the filename, e.g. "actor_train".
        role = "_".join(str(part) for part in (arguments.get("save_file_prefix"), arguments.get("role")) if part)
        export = profiler.export_chrome_trace
        stop = profiler.stop
        dispatch = {"ns": 0}

        def export_chrome_trace(path):
            start = time.perf_counter_ns()
            try:
                if getattr(profiler, "_use_cupti_monitor", False):
                    # Writes the CPU events and queues the GPU merge, which runs on our thread.
                    export(path)
                    take = getattr(profiler, "take_pending_cupti_export", None)
                    observer = take() if take is not None else None
                    if observer is None:  # torch wrote it already
                        _register(path, step, role or None)
                    else:
                        _start_export(_finish_cupti_export, observer, path, step, role or None)
                    return
                # A scheduled profiler starts its next window right after this call
                # and replaces `profiler.profiler`; export the window being saved.
                window = functools.partial(type(profiler).export_chrome_trace, copy.copy(profiler))
                _start_export(_export, window, path, step, role or None)
            finally:
                dispatch["ns"] += time.perf_counter_ns() - start

        def timed_stop(*stop_args, **stop_kwargs):
            dispatch["ns"] = 0
            start_time_ns = time.time_ns()
            try:
                return stop(*stop_args, **stop_kwargs)
            finally:
                try:
                    _emit_stop(start_time_ns, time.time_ns(), profiler, arguments.get("rank"), step, dispatch["ns"])
                except Exception:
                    logger.exception("Failed to record the Torch profiler stop")

        profiler.export_chrome_trace = export_chrome_trace
        profiler.stop = timed_stop
        return profiler

    setattr(get_torch_profiler, _PATCHED, True)
    module.get_torch_profiler = get_torch_profiler
    return True


def patch_finish_hook(module: ModuleType) -> bool:
    """Make ``DistProfiler``'s finish hook (relocation, the user's command) see finished traces."""
    require(module, "DistProfiler")
    profiler_class = module.DistProfiler
    require(profiler_class, "_run_finish_hook")
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
