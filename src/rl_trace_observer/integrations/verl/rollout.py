"""Register ``rollout.name=tokenspeed`` with VERL when it imports its rollout registry.

Only names are registered here; TokenSpeed integration modules are imported when
VERL builds a TokenSpeed replica or adapter.
"""

from types import ModuleType

from rl_trace_observer.integrations.verl.import_hook import when_imported

REPLICA_MODULE = "verl.workers.rollout.replica"
ROLLOUT_NAME = "tokenspeed"
ADAPTER = "rl_trace_observer.integrations.tokenspeed.adapter.TokenSpeedServerAdapter"


def _load_replica():
    from rl_trace_observer.integrations.tokenspeed.replica import TokenSpeedReplica

    return TokenSpeedReplica


def register_rollout(module: ModuleType) -> None:
    from verl.workers.rollout import base

    base._ROLLOUT_REGISTRY.setdefault((ROLLOUT_NAME, "async"), ADAPTER)
    module.RolloutReplicaRegistry.register(ROLLOUT_NAME, _load_replica)


def install() -> None:
    when_imported(REPLICA_MODULE, register_rollout)
