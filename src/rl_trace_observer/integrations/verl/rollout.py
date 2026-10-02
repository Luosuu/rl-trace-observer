"""Register ``rollout.name=tokenspeed`` with VERL when it imports its rollout registry.

Only names are registered here; TokenSpeed integration modules are imported when
VERL builds a TokenSpeed replica, adapter or checkpoint engine. The checkpoint
engine (``checkpoint_engine.backend=tokenspeed``) sends weights to standalone
replicas, for VERL's asynchronous trainers.
"""

from types import ModuleType

from rl_trace_observer.integrations.verl.import_hook import when_imported

REPLICA_MODULE = "verl.workers.rollout.replica"
CHECKPOINT_ENGINE_MODULE = "verl.checkpoint_engine.base"
ROLLOUT_NAME = "tokenspeed"
ADAPTER = "rl_trace_observer.integrations.tokenspeed.adapter.TokenSpeedServerAdapter"


def _load_replica():
    from rl_trace_observer.integrations.tokenspeed.replica import TokenSpeedReplica

    return TokenSpeedReplica


def register_rollout(module: ModuleType) -> None:
    from verl.workers.rollout import base

    base._ROLLOUT_REGISTRY.setdefault((ROLLOUT_NAME, "async"), ADAPTER)
    module.RolloutReplicaRegistry.register(ROLLOUT_NAME, _load_replica)


class _TokenSpeedCheckpointEngine:
    """Stands in for ``TokenSpeedCheckpointEngine`` until VERL builds one."""

    def __new__(cls, *args, **kwargs):
        from rl_trace_observer.integrations.tokenspeed.checkpoint_engine import TokenSpeedCheckpointEngine

        return TokenSpeedCheckpointEngine(*args, **kwargs)

    @classmethod
    def build_topology(cls, *args, **kwargs):
        from rl_trace_observer.integrations.tokenspeed.checkpoint_engine import TokenSpeedCheckpointEngine

        return TokenSpeedCheckpointEngine.build_topology(*args, **kwargs)


def register_checkpoint_engine(module: ModuleType) -> None:
    module.CheckpointEngineRegistry._registry.setdefault(ROLLOUT_NAME, _TokenSpeedCheckpointEngine)


def install() -> None:
    when_imported(REPLICA_MODULE, register_rollout)
    when_imported(CHECKPOINT_ENGINE_MODULE, register_checkpoint_engine)
