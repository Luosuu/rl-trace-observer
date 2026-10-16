"""Patch a module right after it is imported, without importing it ourselves.

The plugin is loaded in every VERL process, so it must not import VERL's
trainer or profiler modules eagerly; it patches them when VERL does.
"""

import importlib.abc
import sys
from collections.abc import Callable
from types import ModuleType


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, callbacks: list[Callable[[ModuleType], None]]):
        self._loader = loader
        self._callbacks = callbacks

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        for callback in self._callbacks:
            callback(module)

    def __getattr__(self, name: str):
        # get_source, get_filename, ...: tracebacks and inspect keep working.
        return getattr(self._loader, name)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Run callbacks right after one module is imported, in the order they were added."""

    def __init__(self, name: str):
        self.name = name
        self.callbacks: list[Callable[[ModuleType], None]] = []

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _PostImportFinder) or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _PostImportLoader(spec.loader, self.callbacks)
                return spec
        return None


def when_imported(name: str, callback: Callable[[ModuleType], None]) -> None:
    """Call ``callback(module)`` now if ``name`` is imported, else right after it is."""
    if name in sys.modules:
        callback(sys.modules[name])
        return
    finder = next(
        (finder for finder in sys.meta_path if isinstance(finder, _PostImportFinder) and finder.name == name), None
    )
    if finder is None:
        finder = _PostImportFinder(name)
        sys.meta_path.insert(0, finder)
    if callback not in finder.callbacks:
        finder.callbacks.append(callback)


def require(owner: object, *names: str) -> None:
    """Fail clearly when VERL lacks what a patch relies on, instead of breaking later."""
    missing = [name for name in names if not hasattr(owner, name)]
    if missing:
        from importlib.metadata import version

        raise RuntimeError(
            f"rl-trace-observer supports verl 0.9.1, but {getattr(owner, '__name__', owner)} of the installed "
            f"verl {version('verl')} has no {', '.join(missing)}"
        )
