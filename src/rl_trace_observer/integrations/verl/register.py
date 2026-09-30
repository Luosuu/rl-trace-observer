import os

from rl_trace_observer.integrations.rl_insight import register_rl_insight_client
from rl_trace_observer.integrations.verl import steps

register_rl_insight_client()
steps.install()

if os.getenv("RL_TRACE_VIZTRACER", "").lower() in {"1", "true", "yes", "on"}:
    from rl_trace_observer.integrations.verl.patch import install
    from rl_trace_observer.integrations.verl.viztracer import VizTracerObserver
    from rl_trace_observer.observer import ObserverRegistry

    if not ObserverRegistry.contains("viztracer"):
        ObserverRegistry.register("viztracer", VizTracerObserver)
    install()
