import logging
import os

from rl_trace_observer.observer import ObserverContext
from rl_trace_observer.output import safe_component, trace_output_dir
from rl_trace_observer.process_record import register_artifact

logger = logging.getLogger(__name__)


class VizTracerObserver:
    """Run VizTracer around VERL's existing profiler window."""

    def __init__(self):
        self._tracer = None

    def on_start(self, context: ObserverContext) -> None:
        role = context.metadata.get("role") or context.save_file_prefix or "unknown"
        allowed_roles = {item.strip() for item in os.getenv("RL_TRACE_VIZTRACER_ROLES", "").split(",") if item.strip()}
        if allowed_roles and str(role) not in allowed_roles:
            return

        from viztracer import VizTracer

        output_dir = trace_output_dir()
        profile_step = context.metadata.get("profile_step", "unknown")
        output_file = output_dir / (
            f"step-{safe_component(profile_step)}-role-{safe_component(role)}-"
            f"rank-{context.rank}-pid-{os.getpid()}.viztracer.json"
        )
        try:
            step = int(profile_step) if str(profile_step).isdigit() else None
            register_artifact(output_dir, "viztracer", output_file, global_step=step)
        except Exception:
            logger.exception("Failed to write the process record for %s", output_file)
        self._tracer = VizTracer(
            output_file=str(output_file),
            min_duration=int(os.getenv("RL_TRACE_VIZTRACER_MIN_DURATION_US", "100")),
            log_async=True,
        )
        self._tracer.start()

    def on_stop(self, context: ObserverContext) -> None:
        if self._tracer is None:
            return
        try:
            self._tracer.stop()
            self._tracer.save()
        finally:
            self._tracer = None
