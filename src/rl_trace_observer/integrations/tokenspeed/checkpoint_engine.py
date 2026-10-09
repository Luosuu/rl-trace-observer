"""``checkpoint_engine.backend=tokenspeed``: weight sync to standalone TokenSpeed replicas.

VERL's asynchronous trainers (``hybrid_engine=False``) run the rollout on GPUs
of its own and send weights through a checkpoint engine: the trainer sends to a
checkpoint-engine worker on each rollout GPU, which hands the weights to its
inference server over CUDA IPC. TokenSpeed has no CUDA-IPC path, so this
backend skips that hop: training rank 0 joins one weight-update group with every
TokenSpeed rank and broadcasts each bucket once (see :mod:`.weight_sync`). The
other training ranks only take part in gathering the full tensors, and the
rollout-side workers receive nothing (their adapters are passive, see
:mod:`.adapter`). The group is created on the first update and kept.

The rl-trace-observer VERL plugin registers the backend in every VERL process.
"""

from collections.abc import AsyncGenerator, Generator
from typing import Any

import ray
import torch
from verl.checkpoint_engine.base import CheckpointEngine
from verl.workers.rollout.utils import ensure_async_iterator

from .server import server_name
from .weight_group import GROUP_PREFIX
from .weight_sync import WeightSender

GROUP_NAME = GROUP_PREFIX + "standalone"


async def _replicas(rollout_world_size: int) -> list[int]:
    """Replica ranks of the TokenSpeed servers that together hold ``rollout_world_size`` ranks."""
    replicas, ranks = [], 0
    while ranks < rollout_world_size:
        replica = len(replicas)
        try:
            server = ray.get_actor(server_name(replica))
        except ValueError as error:
            raise RuntimeError(
                f"Found TokenSpeed servers with {ranks} of {rollout_world_size} rollout ranks; "
                f"{server_name(replica)} does not exist"
            ) from error
        ranks += await server.get_world_size.remote()
        replicas.append(replica)
    return replicas


class TokenSpeedCheckpointEngine(CheckpointEngine):
    def __init__(self, bucket_size: int, is_master: bool = False, **kwargs: Any) -> None:
        self.bucket_size = bucket_size
        self.is_master = is_master
        self.rollout_world_size: int | None = None
        self._sender: WeightSender | None = None

    def prepare(self) -> dict[str, Any]:
        return {}

    @classmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[dict]
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        return (
            {"rollout_world_size": [rollout_world_size] * actor_wg_world_size},
            {"rollout_world_size": [rollout_world_size] * rollout_world_size},
        )

    def init_process_group(self, rollout_world_size: int, **kwargs: Any) -> None:
        self.rollout_world_size = rollout_world_size

    def finalize(self) -> None:
        pass

    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int | None = None
    ) -> dict:
        weights = ensure_async_iterator(weights)
        if not self.is_master:
            # Gathering each full tensor is collective, so every rank iterates.
            async for _ in weights:
                pass
            return {}
        if self._sender is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            device = torch.cuda.current_device() if backend == "nccl" else None
            self._sender = WeightSender(await _replicas(self.rollout_world_size), GROUP_NAME, backend, device)
        count = await self._sender.send(weights, self.bucket_size)
        await self._sender.finish(global_steps)
        print(
            f"[rl-trace-observer] sent {count} tensors to TokenSpeed replicas {self._sender.replicas} "
            f"(step {global_steps})"
        )
        return {}

    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        return
        yield
