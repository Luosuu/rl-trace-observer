"""Create the trainer's side of TokenSpeed's weight-update groups as standalone NCCL communicators.

Since torch 2.14, ``_new_process_group_helper`` splits every new NCCL group
from the default process group when that group is bound to a device. A split
of the trainer's own world would never include TokenSpeed's workers, so
:func:`standalone` creates the group without splitting. TokenSpeed's workers
do the same on their side (our TokenSpeed fork, ``init_weights_update_group``).
"""

import contextlib
from types import ModuleType

# Names of the weight-update groups this package creates.
GROUP_PREFIX = "rl_trace_observer_tokenspeed_"


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
