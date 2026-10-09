"""Patch a module right after it is imported, without importing it ourselves.

The plugin is loaded in every VERL process, so it must not import VERL's
trainer or profiler modules eagerly; it patches them when VERL does.
"""

import importlib.abc
import sys
from collections.abc import Callable
from types import ModuleType


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, callback: Callable[[ModuleType], None]):
        self._loader = loader
        self._callback = callback

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        self._callback(module)

    def __getattr__(self, name: str):
        # get_source, get_filename, ...: tracebacks and inspect keep working.
        return getattr(self._loader, name)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Run a callback right after one module is imported."""

    def __init__(self, name: str, callback: Callable[[ModuleType], None]):
        self.name = name
        self.callback = callback

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        for finder in sys.meta_path:
            if isinstance(finder, _PostImportFinder) or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                    spec.loader = _PostImportLoader(spec.loader, self.callback)
                return spec
        return None


def when_imported(name: str, callback: Callable[[ModuleType], None]) -> None:
    """Call ``callback(module)`` now if ``name`` is imported, else right after it is."""
    if name in sys.modules:
        callback(sys.modules[name])
        return
    # One callback per module keeps the hook simple; that is all the plugin needs.
    installed = [finder for finder in sys.meta_path if isinstance(finder, _PostImportFinder)]
    if any(finder.name == name and finder.callback is not callback for finder in installed):
        raise RuntimeError(f"{name} already has another post-import callback")
    if not any(finder.name == name for finder in installed):
        sys.meta_path.insert(0, _PostImportFinder(name, callback))


def require(owner: object, *names: str) -> None:
    """Fail clearly when VERL lacks what a patch relies on, instead of breaking later."""
    missing = [name for name in names if not hasattr(owner, name)]
    if missing:
        from importlib.metadata import version

        raise RuntimeError(
            f"rl-trace-observer supports verl 0.9.1, but {getattr(owner, '__name__', owner)} of the installed "
            f"verl {version('verl')} has no {', '.join(missing)}"
        )
