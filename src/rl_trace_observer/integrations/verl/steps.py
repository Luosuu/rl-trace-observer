"""Mark each VERL training step on the trainer's RL-Insight timeline.

RL-Insight spans carry no step, so the driver records one ``global_step`` span
per step of VERL's v1 trainer (``PPOTrainer.step``: rollout and training) and of
its one-step-off-policy trainer (``OneStepOffRayTrainer.fit_step``). The merger
uses these spans as the time window of each profile session.

It also tells a TokenSpeed rollout which step it profiles: VERL's v1 trainer
starts rollout profiling without naming the step, and the one-step-off-policy
trainer does not profile its rollout at all. There, step ``n`` trains on the
batch generated during step ``n - 1`` while it generates the batch of step
``n + 1``; the generation that runs during a profiled step is profiled as part
of that step, so its trace shows rollout and training side by side.

The trainers live in heavy modules that the plugin must not import itself, so
the patches are applied when VERL imports them.
"""

import functools
import logging
import time
from types import ModuleType

from rl_trace_observer.context import STEP_MARKER_ATTRIBUTE, STEP_SPAN_NAME, current_run_id
from rl_trace_observer.integrations.verl.import_hook import when_imported

logger = logging.getLogger(__name__)

TRAINER_MODULE = "verl.trainer.ppo.v1.trainer_base"
ONE_STEP_OFF_MODULE = "verl.experimental.one_step_off_policy.ray_trainer"
_PATCHED = "_rl_trace_observer_step_marker"
# Set on a one-step-off-policy trainer while a step runs.
_IN_STEP = "_rl_trace_observer_in_step"


def _emit_step_span(trainer: object, start_time_ns: int, end_time_ns: int, global_step: int | None = None) -> None:
    from verl.utils.tracking import RLInsightLogger

    if not RLInsightLogger.enabled():
        return
    if global_step is None:
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


def _profiles_rollout(trainer, global_step: int) -> bool:
    config = trainer.config
    steps = config.global_profiler.get("steps") or []
    return config.actor_rollout_ref.rollout.name == "tokenspeed" and global_step in steps


def patch_one_step_off_trainer(module: ModuleType) -> bool:
    """Wrap ``module.OneStepOffRayTrainer``'s steps; returns ``False`` if already wrapped."""
    trainer_class = module.OneStepOffRayTrainer
    original_step = trainer_class.fit_step
    if getattr(original_step, _PATCHED, False):
        return False

    @functools.wraps(original_step)
    async def fit_step(self, *args, **kwargs):
        # The step increments global_steps before it returns.
        global_step, start_time_ns = self.global_steps, time.time_ns()
        setattr(self, _IN_STEP, True)
        try:
            return await original_step(self, *args, **kwargs)
        finally:
            setattr(self, _IN_STEP, False)
            try:
                _emit_step_span(self, start_time_ns, time.time_ns(), global_step)
            except Exception:
                logger.exception("Failed to record the global_step span")

    original_generate = trainer_class._async_gen_next_batch

    @functools.wraps(original_generate)
    async def _async_gen_next_batch(self, *args, **kwargs):
        # Runs as a task the step starts; read the step before the first await.
        global_step = self.global_steps
        profile = getattr(self, _IN_STEP, False) and _profiles_rollout(self, global_step)
        if profile:
            # Synchronously: the step blocks the event loop between its
            # `await asyncio.sleep(0)`s, so awaiting here could hold up the
            # generation until the next training phase ends.
            import ray

            servers = [server for replica in self.llm_server_manager.get_replicas() for server in replica.servers]
            ray.get([server.start_profile.remote(global_step=global_step) for server in servers])
        try:
            return await original_generate(self, *args, **kwargs)
        finally:
            if profile:
                await self.llm_server_manager.stop_profile()

    setattr(fit_step, _PATCHED, True)
    trainer_class.fit_step = fit_step
    trainer_class._async_gen_next_batch = _async_gen_next_batch
    return True


def install() -> None:
    when_imported(TRAINER_MODULE, patch_trainer)
    when_imported(ONE_STEP_OFF_MODULE, patch_one_step_off_trainer)
