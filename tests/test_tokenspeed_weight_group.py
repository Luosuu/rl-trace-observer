"""The trainer's weight-update groups must not split from the default process group."""

import types

from rl_trace_observer.integrations.tokenspeed import weight_group


def test_groups_are_created_unbound_and_the_binding_is_restored():
    default = types.SimpleNamespace(bound_device_id="cuda:0")
    c10d = types.SimpleNamespace(is_initialized=lambda: True, _get_default_group=lambda: default)

    with weight_group.standalone(c10d):
        assert default.bound_device_id is None
    assert default.bound_device_id == "cuda:0"


def test_without_a_default_group():
    c10d = types.SimpleNamespace(is_initialized=lambda: False)
    with weight_group.standalone(c10d):
        pass
