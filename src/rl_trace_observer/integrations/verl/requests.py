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
- ``server_id``: the server the call went to.

The span is marked as a point on the request's path, like the server's span
for it, so the merger links them with one flow.

VERL's client lives in a heavy module that the plugin must not import itself,
so the patch is applied when VERL imports it.
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
SPAN_NAME = "rollout_request"
_PATCHED = "_rl_trace_observer_records_requests"

# What the call in progress learned: the server's request id and the server.
_CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar("rl_trace_observer_request", default=None)
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
        call: dict = {}
        token = _CALL.set(call)
        start_time_ns, slot = time.time_ns(), _take_slot()
        try:
            return await original(self, request_id, *args, **kwargs)
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


def install() -> None:
    when_imported(MANAGER_MODULE, patch_server_client)
