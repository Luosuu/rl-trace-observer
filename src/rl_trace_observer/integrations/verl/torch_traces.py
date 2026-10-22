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

By default the profilers record only RecordFunction user scopes: VERL's
``record_function`` markers (``actor_update``, ``mini_batch…``), not every
aten op and autograd function. Those ops are most of the CPU events (hundreds
of thousands per profiled GPU step), and torch keeps and processes each one,
so they cost seconds in every stop and slow the profiled step itself; kernels
still link to the CUDA launches that issued them. ``RL_TRACE_TORCH_CPU_OPS=1``
records the ops too.

The trace is written with torch's Python writer (``use_python_export``):
Kineto's own ``save`` holds the GIL for the whole write, so the training thread
stalls for seconds, inside the stop when the export thread takes the GIL first.

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
import gc
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
CPU_OPS_ENV = "RL_TRACE_TORCH_CPU_OPS"
CUPTI_MONITOR = "cupti_monitor"
STOP_SPAN_NAME = "torch_profiler_stop"
EXPORT_SPAN_NAME = "torch_trace_export"
# Only one profiler is built with a patched torch.profiler.profile at a time.
_construct_lock = threading.Lock()
# Set while one of our profilers enables recording (start, or a scheduled step).
_markers_only = threading.local()

_exports: list[threading.Thread] = []
_exports_lock = threading.Lock()


def _register(path: str, global_step: int | None, role: str | None) -> None:
    try:
        from rl_trace_observer.output import trace_output_dir
        from rl_trace_observer.process_record import register_artifact

        register_artifact(trace_output_dir(), "torch", Path(path), global_step=global_step, role=role)
    except Exception:
        logger.exception("Failed to register the Torch trace %s", path)


def _export(export, path: str, global_step: int | None, role: str | None, lane: str | None = None) -> None:
    start_time_ns = time.time_ns()
    try:
        export(path)
    except Exception:
        logger.exception("Failed to export the Torch trace %s", path)
        return
    _register(path, global_step, role)
    if lane is not None:
        try:
            _emit_export(start_time_ns, lane, path)
        except Exception:
            logger.exception("Failed to record the Torch trace export")


def _emit_export(start_time_ns: int, lane: str, path: str) -> None:
    from verl.utils.tracking import RLInsightLogger

    if RLInsightLogger.enabled():
        attributes = {"state_lane_id": lane, "file": Path(path).name}
        RLInsightLogger.trace_span(
            EXPORT_SPAN_NAME, start_time_ns=start_time_ns, end_time_ns=time.time_ns(), attributes=attributes
        )


class _GcPauses:
    """Garbage collections, in any thread, while the profiler stops: they hold the GIL."""

    def __init__(self):
        self.ns = 0
        self.by_generation = [0, 0, 0]
        self._started: dict[int, int] = {}

    def __call__(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._started[threading.get_ident()] = time.perf_counter_ns()
        elif (started := self._started.pop(threading.get_ident(), None)) is not None:
            self.ns += time.perf_counter_ns() - started
            self.by_generation[info.get("generation", 2)] += 1


def _start_export(target, *args) -> int:
    """Start ``target(*args)`` on an export thread; returns how long starting it took (ns)."""
    start = time.perf_counter_ns()
    # Not a daemon: a normal interpreter exit still finishes the file.
    thread = threading.Thread(target=target, args=args, name="rl-trace-torch-export")
    with _exports_lock:
        _exports.append(thread)
    thread.start()
    return time.perf_counter_ns() - start


def _finish_cupti_export(observer, path: str, global_step: int | None, role: str | None) -> None:
    try:
        # Off the training thread, so without forcing a CUPTI flush (torch's take_pending_cupti_export).
        observer.join(force=False)
    except Exception:
        logger.exception("Failed to export the Torch trace %s", path)
        return
    # The monitor always writes gzipped traces.
    _register(path if path.endswith(".gz") else f"{path}.gz", global_step, role)


def _python_export_kwargs(profiler) -> dict:
    """``use_python_export=True`` where torch has it: Kineto's own writer holds the GIL throughout.

    Kineto's ``save`` keeps the GIL for the whole write (seconds for a GPU
    step), stalling the training thread, or holding up the stop when the export
    thread takes the GIL first. torch's Python writer holds it only to list the
    activities, then gives it up every few milliseconds, writing the same events.
    """
    try:
        parameters = inspect.signature(type(profiler).export_chrome_trace).parameters
    except (TypeError, ValueError):
        return {}
    return {"use_python_export": True} if "use_python_export" in parameters else {}


def _records_cpu_ops() -> bool:
    return os.environ.get(CPU_OPS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _install_scope_filter() -> None:
    """Make torch enable recording for user scopes only while ``_markers_only.active`` is set.

    torch enables RecordFunction recording in ``torch.autograd.profiler``'s
    ``_start_trace`` with every scope; its ``_enable_profiler`` takes the scopes to record.
    """
    import torch.autograd.profiler as autograd_profiler

    enable = autograd_profiler._enable_profiler
    if getattr(enable, _PATCHED, False):
        return
    from torch._C._profiler import RecordScope

    @functools.wraps(enable)
    def _enable_profiler(config, activities, scopes=None):
        if not scopes and getattr(_markers_only, "active", False):
            scopes = {RecordScope.USER_SCOPE}
        return enable(config, activities, scopes or set())

    setattr(_enable_profiler, _PATCHED, True)
    autograd_profiler._enable_profiler = _enable_profiler


def _markers_only_while(method):
    """Run ``method`` (a profiler's start or step) recording only user scopes."""

    @functools.wraps(method)
    def call(*args, **kwargs):
        _markers_only.active = True
        try:
            return method(*args, **kwargs)
        finally:
            _markers_only.active = False

    return call


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


def _emit_stop(start_time_ns: int, end_time_ns: int, profiler, rank, step, dispatch: dict, pauses: _GcPauses) -> None:
    from verl.utils.tracking import RLInsightLogger

    if not RLInsightLogger.enabled():
        return
    total_ns, dispatch_ns = end_time_ns - start_time_ns, dispatch["ns"]
    attributes = {
        "state_lane_id": f"rank_{rank}/profiler",
        "backend": CUPTI_MONITOR if getattr(profiler, "_use_cupti_monitor", False) else "kineto",
        "total_ms": round(total_ns / 1e6, 3),
        "export_dispatch_ms": round(dispatch_ns / 1e6, 3),
        # Parts of the hand-off, and garbage collections in any thread meanwhile.
        "export_copy_ms": round(dispatch["copy_ns"] / 1e6, 3),
        "export_thread_start_ms": round(dispatch["start_ns"] / 1e6, 3),
        "gc_ms": round(pauses.ns / 1e6, 3),
        "gc_collections": list(pauses.by_generation),
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
        _python_export = _python_export_kwargs(profiler)
        dispatch = {"ns": 0, "copy_ns": 0, "start_ns": 0}
        lane = f"rank_{arguments.get('rank')}/export"

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
                        dispatch["start_ns"] += _start_export(_finish_cupti_export, observer, path, step, role or None)
                    return
                # A scheduled profiler starts its next window right after this call
                # and replaces `profiler.profiler`; export the window being saved.
                copied = time.perf_counter_ns()
                window = functools.partial(type(profiler).export_chrome_trace, copy.copy(profiler), **_python_export)
                dispatch["copy_ns"] += time.perf_counter_ns() - copied
                dispatch["start_ns"] += _start_export(_export, window, path, step, role or None, lane)
            finally:
                dispatch["ns"] += time.perf_counter_ns() - start

        def timed_stop(*stop_args, **stop_kwargs):
            dispatch.update(ns=0, copy_ns=0, start_ns=0)
            pauses = _GcPauses()
            gc.callbacks.append(pauses)
            start_time_ns = time.time_ns()
            try:
                return stop(*stop_args, **stop_kwargs)
            finally:
                end_time_ns = time.time_ns()
                gc.callbacks.remove(pauses)
                try:
                    _emit_stop(start_time_ns, end_time_ns, profiler, arguments.get("rank"), step, dispatch, pauses)
                except Exception:
                    logger.exception("Failed to record the Torch profiler stop")

        profiler.export_chrome_trace = export_chrome_trace
        profiler.stop = timed_stop
        if not _records_cpu_ops():
            _install_scope_filter()
            # A scheduled profiler starts recording again in step(), at each window.
            profiler.start = _markers_only_while(profiler.start)
            if hasattr(profiler, "step"):
                profiler.step = _markers_only_while(profiler.step)
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
