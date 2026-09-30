import json
import logging
import os
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rl_trace_observer.output import safe_component, trace_output_dir

logger = logging.getLogger(__name__)

BACKEND_NAME = "rl_trace_observer"
RL_INSIGHT_BACKEND_ENV = "RL_INSIGHT_SERVER_BACKEND"
_TRACE_EVENT_KIND = "trace"
_FORWARD_ENV = "RL_TRACE_FORWARD_TO_RL_INSIGHT"


def _enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


class ChromeTraceJsonlClient:
    """Convert RL-Insight trace events into append-only Chrome events.

    Files are written incrementally because Ray workers may not get a graceful
    shutdown callback. Each line is an independent Chrome trace event that can
    be collected and merged after the profiling run.
    """

    def __init__(self, output_dir: str | Path, delegate: object | None = None):
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        hostname = safe_component(socket.gethostname())
        self.output_path = output_dir / f"rl-insight-{hostname}-pid-{os.getpid()}.chrome.jsonl"
        self._delegate = delegate
        self._write_lock = threading.Lock()

    def apply_event(self, event: dict[str, Any]) -> None:
        """Persist trace events locally and optionally forward every event."""
        try:
            chrome_event = self.to_chrome_event(event)
            if chrome_event is not None:
                encoded = json.dumps(chrome_event, separators=(",", ":"), sort_keys=True)
                with self._write_lock, self.output_path.open("a", encoding="utf-8") as output:
                    output.write(encoded)
                    output.write("\n")
        except Exception:
            logger.exception("Failed to persist an RL-Insight trace event")

        if self._delegate is not None:
            try:
                self._delegate.apply_event(event)
            except Exception:
                logger.exception("Failed to forward an RL-Insight event to the original backend")

    @staticmethod
    def to_chrome_event(event: dict[str, Any]) -> dict[str, Any] | None:
        if event.get("kind") != _TRACE_EVENT_KIND:
            return None

        start_time_ns = int(event["start_time_ns"])
        end_time_ns = int(event["end_time_ns"])
        attributes = dict(event.get("attributes") or {})
        process_id = attributes.get("process_id", os.getpid())
        lane_id = attributes.get("state_lane_id", "rl_insight")
        return {
            "name": str(event["name"]),
            "cat": "rl_insight.state",
            "ph": "X",
            "ts": start_time_ns / 1_000,
            "dur": max(0, end_time_ns - start_time_ns) / 1_000,
            "pid": process_id,
            "tid": lane_id,
            "args": attributes,
        }


def create_rl_trace_observer_client(config: object) -> ChromeTraceJsonlClient:
    """RL-Insight monitor client factory registered as ``rl_trace_observer``.

    With ``RL_TRACE_FORWARD_TO_RL_INSIGHT`` set, every event is also forwarded
    to RL-Insight's built-in Ray monitor client.
    """
    delegate = None
    if _enabled(_FORWARD_ENV):
        from rl_insight.client.ray_monitor_client import create_ray_monitor_client

        delegate = create_ray_monitor_client(config)

    return ChromeTraceJsonlClient(output_dir=trace_output_dir(), delegate=delegate)


def _supports_backend_env() -> bool:
    from rl_insight.utils.constants import MonitorEnv

    return getattr(MonitorEnv, "SERVER_BACKEND", None) == RL_INSIGHT_BACKEND_ENV


def register_rl_insight_client(register: Callable[[str, Callable], None] | None = None) -> None:
    """Register the backend and make it RL-Insight's default in this process.

    VERL workers lazily call ``rl_insight.init()`` without the trainer config,
    so the backend must also be selected through ``RL_INSIGHT_SERVER_BACKEND``.
    It defaults to this backend unless the user already set it.

    Args:
        register: RL-Insight's ``register_monitor_client``; injected by tests.

    Raises:
        RuntimeError: The installed RL-Insight does not support
            ``RL_INSIGHT_SERVER_BACKEND``.
    """
    if register is None:
        from rl_insight.client.base import register_monitor_client

        if not _supports_backend_env():
            raise RuntimeError(
                f"The installed RL-Insight does not support {RL_INSIGHT_BACKEND_ENV}; install the "
                "pinned fork with `uv sync`"
            )
        register = register_monitor_client

    register(BACKEND_NAME, create_rl_trace_observer_client)
    os.environ.setdefault(RL_INSIGHT_BACKEND_ENV, BACKEND_NAME)
