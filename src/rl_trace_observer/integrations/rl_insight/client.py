import json
import logging
import os
import re
import socket
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BACKEND_NAME = "rl_trace_observer"
_TRACE_EVENT_KIND = "trace"


def _enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


def _safe_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


class ChromeTraceJsonlClient:
    """Convert RL-Insight trace events into append-only Chrome events.

    Files are written incrementally because Ray workers may not get a graceful
    shutdown callback. Each line is an independent Chrome trace event that can
    be collected and merged after the profiling run.
    """

    def __init__(self, output_dir: str | Path, delegate: object | None = None):
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        hostname = _safe_component(socket.gethostname())
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
            self._delegate.apply_event(event)

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
    """RL-Insight monitor client factory registered by the external module."""
    delegate = None
    if _enabled("RL_TRACE_FORWARD_TO_RL_INSIGHT"):
        from rl_insight.client.ray_monitor_client import create_ray_monitor_client

        delegate = create_ray_monitor_client(config)

    output_dir = os.getenv("RL_TRACE_OUTPUT_DIR", "rl_trace_outputs")
    return ChromeTraceJsonlClient(output_dir=output_dir, delegate=delegate)


def register_rl_insight_client(register: Callable[[str, Callable], None] | None = None) -> bool:
    """Register the custom backend without importing RL-Insight at package import.

    Returns ``False`` when RL-Insight is unavailable, allowing the optional
    actor VizTracer integration to remain usable on its own.
    """
    if register is None:
        try:
            from rl_insight.client.base import register_monitor_client
        except ImportError:
            logger.warning("RL-Insight is unavailable; semantic state collection is disabled")
            return False
        register = register_monitor_client

    register(BACKEND_NAME, create_rl_trace_observer_client)
    return True
