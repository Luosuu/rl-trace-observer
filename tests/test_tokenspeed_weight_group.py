"""TokenSpeed weight-update groups must not split from the default process group."""

import os
import subprocess
import sys
import types

from rl_trace_observer.integrations.tokenspeed import weight_group


def _fake_c10d():
    default = types.SimpleNamespace(bound_device_id="cuda:0")
    seen = []

    def helper(*args, group_name=None, **kwargs):
        seen.append((group_name, default.bound_device_id))
        return "group", None

    module = types.SimpleNamespace(
        is_initialized=lambda: True,
        _get_default_group=lambda: default,
        _new_process_group_helper=helper,
    )
    return module, default, seen


def test_only_weight_groups_are_created_standalone():
    c10d, default, seen = _fake_c10d()
    weight_group._patch(c10d)
    weight_group._patch(c10d)  # idempotent

    c10d._new_process_group_helper(2, 1, [], "nccl", None, group_name=weight_group.GROUP_PREFIX + "0")
    c10d._new_process_group_helper(2, 1, [], "nccl", None, group_name="tp_group")

    assert seen == [(weight_group.GROUP_PREFIX + "0", None), ("tp_group", "cuda:0")]
    assert default.bound_device_id == "cuda:0"


def test_sitecustomize_patches_server_processes():
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(weight_group.SITE_DIR), os.environ.get("PYTHONPATH", "")])}
    code = (
        "from torch.distributed import distributed_c10d as c10d; "
        f"print(getattr(c10d._new_process_group_helper, {weight_group._PATCHED!r}, False))"
    )
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True"
