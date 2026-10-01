"""Create TokenSpeed's weight-update groups as standalone NCCL communicators.

TokenSpeed's workers join the trainer's weight-update group with torch's
``_new_process_group_helper``, which since torch 2.14 splits every new NCCL
group from the default process group when that group is bound to a device.
TokenSpeed binds its default group, so the "weight-update group" becomes a
split of the worker's own world that never includes the trainer, and the first
broadcast waits forever. :func:`standalone` creates the group without splitting
for groups named with :data:`GROUP_PREFIX`; :func:`install` applies it inside
TokenSpeed's processes through ``sitecustomize`` (see :data:`SITE_DIR`),
without changing TokenSpeed.
"""

import contextlib
import functools
from pathlib import Path
from types import ModuleType

GROUP_PREFIX = "rl_trace_observer_tokenspeed_"
# Prepended to PYTHONPATH of `tokenspeed serve`; its sitecustomize calls install().
SITE_DIR = Path(__file__).resolve().parent / "_site"
_PATCHED = "_rl_trace_observer_standalone_weight_groups"


@contextlib.contextmanager
def standalone(c10d: ModuleType):
    """Within this block, new process groups do not split from the default group."""
    default = c10d._get_default_group() if c10d.is_initialized() else None
    bound = default.bound_device_id if default is not None else None
    if bound is not None:
        default.bound_device_id = None
    try:
        yield
    finally:
        if bound is not None:
            default.bound_device_id = bound


def _patch(c10d: ModuleType) -> None:
    original = c10d._new_process_group_helper
    if getattr(original, _PATCHED, False):
        return

    @functools.wraps(original)
    def helper(*args, **kwargs):
        if not str(kwargs.get("group_name") or "").startswith(GROUP_PREFIX):
            return original(*args, **kwargs)
        with standalone(c10d):
            return original(*args, **kwargs)

    setattr(helper, _PATCHED, True)
    c10d._new_process_group_helper = helper


def install() -> None:
    from rl_trace_observer.integrations.verl.import_hook import when_imported

    when_imported("torch.distributed.distributed_c10d", _patch)
