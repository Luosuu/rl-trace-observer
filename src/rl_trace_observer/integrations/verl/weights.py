"""Record each weight update of the rollout servers on the trainer's timeline.

Every VERL trainer updates the rollout servers' weights through
``CheckpointEngineManager.update_weights(global_steps)``, and VERL tags each
generated sample with the ``global_steps`` of the last update its server got
(the request's ``weight_version``). This patch records one ``weight_update``
RL-Insight span per update, on the trainer's ``trainer/weights`` lane, with
that version as ``weight_version``; the merger turns these spans into a
``rollout weight_version`` counter track, so the timeline shows which weights
the servers held at any moment.

An update without ``global_steps`` (VERL restoring the same weights, e.g.
after a colocated reward model used the GPUs) is not a new version and is not
recorded.

``CheckpointEngineManager`` lives in a heavy module that the plugin must not
import itself, so the patch is applied when VERL imports it.
"""

import functools
import logging
import time
from types import ModuleType

from rl_trace_observer.context import WEIGHT_UPDATE_ATTRIBUTE, WEIGHT_VERSION_ATTRIBUTE, current_run_id
from rl_trace_observer.integrations.verl.import_hook import require, when_imported

logger = logging.getLogger(__name__)

CHECKPOINT_ENGINE_MODULE = "verl.checkpoint_engine.base"
SPAN_NAME = "weight_update"
_PATCHED = "_rl_trace_observer_records_weight_updates"


def _emit(start_time_ns: int, global_steps: int) -> None:
    from verl.utils.tracking import RLInsightLogger

    if not RLInsightLogger.enabled():
        return
    attributes = {
        "state_lane_id": "trainer/weights",
        WEIGHT_VERSION_ATTRIBUTE: int(global_steps),
        WEIGHT_UPDATE_ATTRIBUTE: True,
    }
    if run_id := current_run_id():
        attributes["run_id"] = run_id
    RLInsightLogger.trace_span(
        SPAN_NAME, start_time_ns=start_time_ns, end_time_ns=time.time_ns(), attributes=attributes
    )


def patch_checkpoint_engine_manager(module: ModuleType) -> bool:
    """Wrap ``module.CheckpointEngineManager.update_weights``; returns ``False`` if already wrapped."""
    require(module, "CheckpointEngineManager")
    manager_class = module.CheckpointEngineManager
    require(manager_class, "update_weights")
    original = manager_class.update_weights
    if getattr(original, _PATCHED, False):
        return False
    # VERL's ``auto_await`` decides from its caller's frame whether to return
    # the coroutine or run it, so wrap the coroutine function inside it and
    # decorate the wrapper the same way.
    require(original, "__wrapped__")
    from verl.utils.ray_utils import auto_await

    update = original.__wrapped__

    @functools.wraps(update)
    async def update_weights(self, global_steps: int = None, *args, **kwargs):
        start_time_ns = time.time_ns()
        result = await update(self, global_steps, *args, **kwargs)
        if global_steps is not None:
            try:
                _emit(start_time_ns, global_steps)
            except Exception:
                logger.exception("Failed to record the weight_update span")
        return result

    wrapped = auto_await(update_weights)
    setattr(wrapped, _PATCHED, True)
    manager_class.update_weights = wrapped
    return True


def install() -> None:
    when_imported(CHECKPOINT_ENGINE_MODULE, patch_checkpoint_engine_manager)
