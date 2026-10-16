"""Record each rollout request VERL's agent loop sends, for request-level correlation.

VERL's agent loop calls ``LLMServerClient.generate(request_id, ...)``
once per model call (the fully-async policy's client once per resumed
segment of a partial rollout). That ``request_id`` only picks a server (a sticky session
across the turns of one trajectory); the server is sent a fresh id per call
(``_vllm_request_id``), which the server uses as the request's id end to end
(TokenSpeed keeps it from HTTP to its scheduler). This patch records one
``rollout_request`` RL-Insight span per call on the agent loop worker's
timeline, with:

- ``request_id``: the id the server was sent;
- ``trajectory_id``: the agent loop's id, shared by the turns of a trajectory;
- ``server_id``: the server the call went to;
- ``weight_version``: the weights it was generated with, as VERL tags its output
  (``global_steps`` of the last weight update the server received), with
  ``min_weight_version`` and ``max_weight_version`` when they differ (weights
  updated during a partial rollout);
- ``batch_step``, ``sample_index``, ``rollout_n``, ``validate``: the trajectory,
  as VERL's agent loop knows it (``batch_step`` is the trainer's
  ``global_steps`` when it asked for the batch; see ``steps`` for which step
  trains on it).

The span is marked as a point on the request's path, like the server's span
for it, so the merger links them with one flow.

VERL's client and agent loop live in heavy modules that the plugin must not
import itself, so the patches are applied when VERL imports them.
"""

import contextvars
import functools
import logging
import threading
import time
from types import ModuleType

from rl_trace_observer.context import REQUEST_FLOW_ATTRIBUTE, REQUEST_ID_ATTRIBUTE
from rl_trace_observer.integrations.verl.import_hook import require, when_imported

logger = logging.getLogger(__name__)

MANAGER_MODULE = "verl.workers.rollout.llm_server"
AGENT_LOOP_MODULE = "verl.experimental.agent_loop.agent_loop"
SPAN_NAME = "rollout_request"
_PATCHED = "_rl_trace_observer_records_requests"

# What the call in progress learned: the server's request id and the server.
_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar("rl_trace_observer_request", default=None)
# The trajectory VERL's agent loop is running (its ``trajectory`` argument).
_TRAJECTORY: contextvars.ContextVar[dict | None] = contextvars.ContextVar("rl_trace_observer_trajectory", default=None)
_TRAJECTORY_FIELDS = {
    "step": "batch_step",
    "sample_index": "sample_index",
    "rollout_n": "rollout_n",
    "validate": "validate",
}
# Concurrent calls get their own lanes, so their spans never overlap.
_busy_slots: set[int] = set()
_slots_lock = threading.Lock()


def _take_slot() -> int:
    with _slots_lock:
        slot = next(index for index in range(len(_busy_slots) + 1) if index not in _busy_slots)
        _busy_slots.add(slot)
        return slot


def _release_slot(slot: int) -> None:
    with _slots_lock:
        _busy_slots.discard(slot)


def _plain(value):
    """A JSON-friendly scalar (VERL passes numpy integers)."""
    return value.item() if hasattr(value, "item") else value


def _emit(start_time_ns: int, call: dict, trajectory_id: object, slot: int) -> None:
    from verl.utils.tracking import RLInsightLogger

    if not RLInsightLogger.enabled() or call.get("request_id") is None:
        return
    attributes = {
        "state_lane_id": f"agent_loop/slot_{slot}",
        REQUEST_ID_ATTRIBUTE: str(call["request_id"]),
        "trajectory_id": str(trajectory_id),
        REQUEST_FLOW_ATTRIBUTE: True,
    }
    if call.get("server_id") is not None:
        attributes["server_id"] = str(call["server_id"])
    extra = getattr(call.get("output"), "extra_fields", None) or {}
    if extra.get("global_steps") is not None:
        attributes["weight_version"] = _plain(extra["global_steps"])
    versions = {_plain(extra.get(key)) for key in ("min_global_steps", "max_global_steps")} - {None}
    if len(versions) > 1:
        attributes["min_weight_version"], attributes["max_weight_version"] = min(versions), max(versions)
    for key, attribute in _TRAJECTORY_FIELDS.items():
        if (value := (call.get("trajectory") or {}).get(key)) is not None:
            attributes[attribute] = _plain(value)
    RLInsightLogger.trace_span(
        SPAN_NAME, start_time_ns=start_time_ns, end_time_ns=time.time_ns(), attributes=attributes
    )


def patch_server_client(module: ModuleType) -> bool:
    """Wrap ``module.LLMServerClient.generate``; returns ``False`` if already wrapped."""
    require(module, "LLMServerClient")
    manager_class = module.LLMServerClient
    require(manager_class, "generate", "_vllm_request_id", "_acquire_server")
    original = manager_class.generate
    if getattr(original, _PATCHED, False):
        return False
    original_request_id = manager_class._vllm_request_id
    original_acquire = manager_class._acquire_server

    @functools.wraps(original_request_id)
    def _vllm_request_id(self, *args, **kwargs):
        request_id = original_request_id(self, *args, **kwargs)
        if (call := _CALL.get()) is not None:
            call["request_id"] = request_id
        return request_id

    @functools.wraps(original_acquire)
    async def _acquire_server(self, *args, **kwargs):
        acquired = await original_acquire(self, *args, **kwargs)
        if (call := _CALL.get()) is not None and isinstance(acquired, tuple) and acquired:
            call["server_id"] = acquired[0]
        return acquired

    @functools.wraps(original)
    async def generate(self, request_id, *args, **kwargs):
        call: dict = {"trajectory": _TRAJECTORY.get()}
        token = _CALL.set(call)
        start_time_ns, slot = time.time_ns(), _take_slot()
        try:
            call["output"] = await original(self, request_id, *args, **kwargs)
            return call["output"]
        finally:
            _CALL.reset(token)
            _release_slot(slot)
            try:
                _emit(start_time_ns, call, request_id, slot)
            except Exception:
                logger.exception("Failed to record the rollout request span")

    setattr(generate, _PATCHED, True)
    manager_class.generate = generate
    manager_class._vllm_request_id = _vllm_request_id
    manager_class._acquire_server = _acquire_server
    return True


def patch_agent_loop(module: ModuleType) -> bool:
    """Wrap ``module.AgentLoopWorker._run_agent_loop`` to expose its trajectory; ``False`` if already wrapped."""
    require(module, "AgentLoopWorker")
    worker_class = module.AgentLoopWorker
    require(worker_class, "_run_agent_loop")
    original = worker_class._run_agent_loop
    if getattr(original, _PATCHED, False):
        return False

    @functools.wraps(original)
    async def _run_agent_loop(self, sampling_params, trajectory, *args, **kwargs):
        token = _TRAJECTORY.set(trajectory if isinstance(trajectory, dict) else None)
        try:
            return await original(self, sampling_params, trajectory, *args, **kwargs)
        finally:
            _TRAJECTORY.reset(token)

    setattr(_run_agent_loop, _PATCHED, True)
    worker_class._run_agent_loop = _run_agent_loop
    return True


def install() -> None:
    when_imported(MANAGER_MODULE, patch_server_client)
    when_imported(AGENT_LOOP_MODULE, patch_agent_loop)
