import json
import logging
import os
import re
import socket
import threading
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BACKEND_NAME = "rl_trace_observer"
DEFAULT_BACKEND = "ray"
_TRACE_EVENT_KIND = "trace"
_FORWARD_ENV = "RL_TRACE_FORWARD_TO_RL_INSIGHT"
_CAPTURE_DEFAULT_ENV = "RL_TRACE_CAPTURE_DEFAULT_BACKEND"


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


def create_rl_trace_observer_client(
    config: object, original_factory: Callable[[object], object] | None = None
) -> ChromeTraceJsonlClient:
    """Create a local JSONL client that optionally tees into the original backend.

    Args:
        config: Merged RL-Insight monitor config passed by ``create_monitor_client``.
        original_factory: RL-Insight's own client factory, used as the forwarding
            delegate. When omitted, the built-in Ray monitor client is imported.
    """
    delegate = None
    if _enabled(_FORWARD_ENV):
        if original_factory is None:
            from rl_insight.client.ray_monitor_client import create_ray_monitor_client

            original_factory = create_ray_monitor_client
        delegate = original_factory(config)

    output_dir = os.getenv("RL_TRACE_OUTPUT_DIR", "rl_trace_outputs")
    return ChromeTraceJsonlClient(output_dir=output_dir, delegate=delegate)


class _TeeFactory:
    """Registry entry that remembers the RL-Insight factory it replaced."""

    def __init__(self, original_factory: Callable[[object], object] | None):
        self.original_factory = original_factory

    def __call__(self, config: object) -> ChromeTraceJsonlClient:
        return create_rl_trace_observer_client(config, original_factory=self.original_factory)


def _default_registry() -> MutableMapping[str, Callable] | None:
    try:
        # Importing ``base`` runs ``rl_insight.client.__init__`` first, which
        # registers the built-in Ray factory that is captured below.
        from rl_insight.client.base import MONITOR_CLIENT_REGISTRY
    except ImportError:
        return None
    return MONITOR_CLIENT_REGISTRY


def register_rl_insight_client(registry: MutableMapping[str, Callable] | None = None) -> bool:
    """Register the custom backend and capture RL-Insight's default backend.

    VERL workers lazily call ``rl_insight.init()`` without the trainer config,
    so they always select the default ``ray`` backend. The default factory is
    therefore replaced by a tee that writes local JSONL and, when
    ``RL_TRACE_FORWARD_TO_RL_INSIGHT`` is set, forwards to the original Ray
    factory. Set ``RL_TRACE_CAPTURE_DEFAULT_BACKEND=0`` to leave it untouched.

    Registration is idempotent: an already installed tee is never wrapped again.

    Returns:
        ``False`` when RL-Insight is unavailable, allowing the optional actor
        VizTracer integration to remain usable on its own.
    """
    if registry is None:
        registry = _default_registry()
        if registry is None:
            logger.warning("RL-Insight is unavailable; semantic state collection is disabled")
            return False

    current_default = registry.get(DEFAULT_BACKEND)
    if isinstance(current_default, _TeeFactory):
        tee_factory = current_default
    else:
        if current_default is None:
            logger.warning("RL-Insight has no %r backend registered; capturing it locally only", DEFAULT_BACKEND)
        tee_factory = _TeeFactory(current_default)
        if os.getenv(_CAPTURE_DEFAULT_ENV, "1").lower() not in {"0", "false", "no", "off"}:
            registry[DEFAULT_BACKEND] = tee_factory
    registry[BACKEND_NAME] = tee_factory
    return True
