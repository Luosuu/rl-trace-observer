"""Mark each VERL training step on the trainer's RL-Insight timeline.

RL-Insight spans carry no step, so the driver records one ``global_step`` span
per step of VERL's v1 trainer (``PPOTrainer.step``: rollout and training). The
merger uses these spans as the time window of each profile session.

It also tells a TokenSpeed rollout which step it profiles: VERL starts rollout
profiling without naming the step.

``PPOTrainer`` lives in a heavy module that the plugin must not import itself,
so the patch is applied when VERL imports it.
"""

import functools
import logging
import time
from types import ModuleType

from rl_trace_observer.context import STEP_MARKER_ATTRIBUTE, STEP_SPAN_NAME, current_run_id
from rl_trace_observer.integrations.verl.import_hook import when_imported

logger = logging.getLogger(__name__)

TRAINER_MODULE = "verl.trainer.ppo.v1.trainer_base"
_PATCHED = "_rl_trace_observer_step_marker"


def _emit_step_span(trainer: object, start_time_ns: int, end_time_ns: int) -> None:
    from verl.utils.tracking import RLInsightLogger

    if not RLInsightLogger.enabled():
        return
    global_step = getattr(trainer, "global_steps", None)
    if global_step is None:
        return
    attributes = {"state_lane_id": "trainer", "global_step": int(global_step), STEP_MARKER_ATTRIBUTE: True}
    if run_id := current_run_id():
        attributes["run_id"] = run_id
    RLInsightLogger.trace_span(
        STEP_SPAN_NAME, start_time_ns=start_time_ns, end_time_ns=end_time_ns, attributes=attributes
    )
    try:
        from rl_trace_observer.output import trace_output_dir
        from rl_trace_observer.process_record import set_process_role

        set_process_role(trace_output_dir(), "trainer")
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

    original_start_profiling = trainer_class._start_rollout_profiling

    @functools.wraps(original_start_profiling)
    def _start_rollout_profiling(self):
        if self.config.actor_rollout_ref.rollout.name != "tokenspeed":
            return original_start_profiling(self)
        for manager in self._rollout_server_managers():
            manager.start_profile(global_step=self.global_steps)

    setattr(step, _PATCHED, True)
    trainer_class.step = step
    trainer_class._start_rollout_profiling = _start_rollout_profiling
    return True


def install() -> None:
    when_imported(TRAINER_MODULE, patch_trainer)
