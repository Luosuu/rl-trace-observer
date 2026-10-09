"""The rollout side of each training worker: wakes, sleeps and updates its TokenSpeed replica.

In hybrid mode, weights travel over TokenSpeed's own RL weight-sync API (see
:mod:`.weight_sync`). TokenSpeed has no CUDA-IPC path, and NCCL rejects two
ranks of one group on the same GPU, so a hybrid replica cannot receive from the
trainer ranks that share its GPUs. Replica ``k`` therefore receives from the
first training rank of replica ``k + 1``: every training rank holds the full
tensors anyway, since gathering them is collective. With a single replica this
needs a backend that allows it (``gloo``, as in CPU tests).

With standalone replicas (an asynchronous trainer), the ``tokenspeed``
checkpoint engine sends the weights from the trainer (see
:mod:`.checkpoint_engine`), and this adapter, created by VERL in a
checkpoint-engine worker on each rollout GPU, has nothing to do.
"""

import asyncio
import logging
import os
from collections.abc import Generator
from typing import Any

import torch
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import ensure_async_iterator

from .weight_group import GROUP_PREFIX
from .weight_sync import WeightSender

logger = logging.getLogger(__name__)

GROUP_NAME = GROUP_PREFIX + "{replica}"
# Checkpoint-engine backend of standalone TokenSpeed replicas (see .checkpoint_engine).
STANDALONE_BACKEND = "tokenspeed"


class TokenSpeedServerAdapter(BaseRollout):
    def __init__(self, config, model_config, device_mesh, *args, replica_rank: int = -1, **kwargs):
        super().__init__(config, model_config, device_mesh)
        self.rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.replica_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self.num_replicas = max(1, world_size // self.replica_size)
        self.replica_rank = self.rank // self.replica_size if replica_rank == -1 else replica_rank
        self.is_leader_rank = self.rank % self.replica_size == 0
        self.passive = self.config.checkpoint_engine.backend == STANDALONE_BACKEND
        self.backend = "nccl" if torch.cuda.is_available() and not self.passive else "gloo"
        # The trainer's GPU. CUDA's current device is per thread, and the
        # group is created and used from worker threads.
        self.device = torch.cuda.current_device() if self.backend == "nccl" else None
        # The replica this rank sends weights to, if any (see the module docstring).
        self.target_replica = None
        if self.is_leader_rank and not self.passive:
            self.target_replica = (self.replica_rank - 1) % self.num_replicas
            if self.num_replicas == 1 and self.backend == "nccl":
                raise ValueError(
                    "TokenSpeed rollout needs at least two replicas on GPUs: a replica receives weights over "
                    "NCCL from a training rank on other GPUs. Use more GPUs or a smaller tensor_model_parallel_size."
                )
        targets = [] if self.target_replica is None else [self.target_replica]
        self._sender = WeightSender(targets, GROUP_NAME.format(replica=self.target_replica), self.backend, self.device)

    async def resume(self, tags: list[str]):
        if self.is_leader_rank and self.config.free_cache_engine:
            await self._sender.post(self.replica_rank, "/resume_memory_occupation", {"tags": tags})

    async def release(self):
        if self.is_leader_rank and self.config.free_cache_engine:
            await self._sender.post(self.replica_rank, "/release_memory_occupation", {"tags": ["kv_cache", "weights"]})

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
        wire_format: str = "named_tensors",
        **kwargs: Any,
    ):
        if self.passive:
            return
        if wire_format != "named_tensors":
            raise NotImplementedError(f"TokenSpeed rollout does not support wire_format={wire_format!r}")
        if kwargs.get("peft_config") is not None:
            raise NotImplementedError("TokenSpeed rollout does not support LoRA adapters")
        # The receiving replica must have resumed its weights, which its own
        # leader does just before this call.
        if torch.distributed.is_initialized():
            await asyncio.to_thread(torch.distributed.barrier)

        weights = ensure_async_iterator(weights)
        if self.target_replica is None:
            # Every rank iterates: gathering each full tensor is collective.
            async for _ in weights:
                pass
            return
        bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
        count = await self._sender.send(weights, bucket_bytes)
        await self._sender.finish(global_steps)
        replica = self.target_replica
        print(f"[rl-trace-observer] sent {count} tensors to TokenSpeed replica {replica} (step {global_steps})")
