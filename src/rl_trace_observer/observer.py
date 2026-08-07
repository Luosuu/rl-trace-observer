import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObserverContext:
    """Immutable view of one VERL profiler lifecycle."""

    rank: int
    tool: str | None
    config: object
    tool_config: object | None
    save_file_prefix: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_profiler(cls, profiler: object, metadata: Mapping[str, Any]) -> "ObserverContext":
        return cls(
            rank=profiler.rank,
            tool=getattr(profiler, "_tool", None),
            config=profiler.config,
            tool_config=getattr(profiler, "tool_config", None),
            save_file_prefix=getattr(profiler, "save_file_prefix", None),
            metadata=MappingProxyType(dict(metadata)),
        )


class ProfilerObserver(Protocol):
    def on_start(self, context: ObserverContext) -> None: ...

    def on_stop(self, context: ObserverContext) -> None: ...


ObserverFactory = Callable[[], ProfilerObserver]


class ObserverRegistry:
    """Registry owned by the independent integration package."""

    _factories: dict[str, ObserverFactory] = {}

    @classmethod
    def register(cls, name: str, factory: ObserverFactory) -> None:
        if not name:
            raise ValueError("Observer name must not be empty")
        if not callable(factory):
            raise TypeError("Observer factory must be callable")
        if name in cls._factories:
            raise ValueError(f"Observer {name!r} is already registered")
        cls._factories[name] = factory

    @classmethod
    def contains(cls, name: str) -> bool:
        return name in cls._factories

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._factories.pop(name, None)

    @classmethod
    def create_observers(cls) -> list[tuple[str, ProfilerObserver]]:
        observers = []
        for name, factory in tuple(cls._factories.items()):
            try:
                observers.append((name, factory()))
            except Exception:
                logger.exception("Failed to create observer %r; skipping it", name)
        return observers
