"""The rollout side of each training worker: wakes, sleeps and updates its TokenSpeed replica.

Weights travel over TokenSpeed's own RL weight-sync API: the trainer creates a
``torch.distributed`` group with the replica's TP ranks
(``/init_weights_update_group``) and broadcasts each bucket of full tensors
from rank 0 while TokenSpeed receives it (``/update_weights_from_distributed``).
TokenSpeed has no CUDA-IPC path, and NCCL rejects two ranks of one group on the
same GPU, so a hybrid replica cannot receive from the trainer ranks that share
its GPUs. Replica ``k`` therefore receives from the first training rank of
replica ``k + 1``: every training rank holds the full tensors anyway, since
gathering them is collective. With a single replica this needs a backend that
allows it (``gloo``, as in CPU tests).
"""

import asyncio
import logging
import os
from collections.abc import Generator
from typing import Any

import aiohttp
import ray
import torch
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import ensure_async_iterator

from .server import _free_port, server_name

logger = logging.getLogger(__name__)

GROUP_NAME = "rl_trace_observer_tokenspeed_{replica}"


def _join_group(master_address: str, master_port: int, world_size: int, group_name: str, backend: str):
    """Create the weight-update group as its rank 0, the way TokenSpeed's workers join it."""
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    store, rank, world_size = next(
        rendezvous(f"tcp://{master_address}:{master_port}", 0, world_size, timeout=default_pg_timeout)
    )
    store.set_timeout(default_pg_timeout)
    group, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend(backend),
        PrefixStore(group_name, store),
        group_name=group_name,
        backend_options=None,
        timeout=default_pg_timeout,
    )
    _world.pg_group_ranks[group] = {i: i for i in range(world_size)}
    return group


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
        self.backend = "nccl" if torch.cuda.is_available() else "gloo"
        # The replica this rank sends weights to, if any (see the module docstring).
        self.target_replica = None
        if self.is_leader_rank:
            self.target_replica = (self.replica_rank - 1) % self.num_replicas
            if self.num_replicas == 1 and self.backend == "nccl":
                raise ValueError(
                    "TokenSpeed rollout needs at least two replicas on GPUs: a replica receives weights over "
                    "NCCL from a training rank on other GPUs. Use more GPUs or a smaller tensor_model_parallel_size."
                )
        self._addresses: dict[int, str] = {}
        self._session: aiohttp.ClientSession | None = None
        self._group = None

    async def _url(self, replica: int, path: str) -> str:
        if replica not in self._addresses:
            server = ray.get_actor(server_name(replica))
            address, port = await server.get_server_address.remote()
            self._addresses[replica] = f"http://{address}:{port}"
        return self._addresses[replica] + path

    async def _post(self, replica: int, path: str, body: dict | None = None) -> dict:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=30))
        async with self._session.post(await self._url(replica, path), json=body or {}) as resp:
            payload = await resp.json(content_type=None)
            if resp.status != 200 or (isinstance(payload, dict) and payload.get("success") is False):
                raise RuntimeError(f"TokenSpeed replica {replica} {path} failed ({resp.status}): {payload}")
            return payload

    async def resume(self, tags: list[str]):
        if self.is_leader_rank and self.config.free_cache_engine:
            await self._post(self.replica_rank, "/resume_memory_occupation", {"tags": tags})

    async def release(self):
        if self.is_leader_rank and self.config.free_cache_engine:
            await self._post(self.replica_rank, "/release_memory_occupation", {"tags": ["kv_cache", "weights"]})

    async def _ensure_group(self, replica: int) -> None:
        if self._group is not None:
            return
        address = ray.util.get_node_ip_address().strip("[]")
        port = _free_port()
        world_size = 1 + self.replica_size
        group_name = GROUP_NAME.format(replica=replica)
        body = {
            "master_address": address,
            "master_port": port,
            "rank_offset": 1,
            "world_size": world_size,
            "group_name": group_name,
            "backend": self.backend,
        }
        joined = asyncio.create_task(self._post(replica, "/init_weights_update_group", body))
        self._group = await asyncio.to_thread(_join_group, address, port, world_size, group_name, self.backend)
        await joined
        logger.info("Rank %d sends weights to TokenSpeed replica %d (%s)", self.rank, replica, self.backend)

    async def _send_bucket(self, replica: int, bucket: list[tuple[str, torch.Tensor]]) -> None:
        body = {
            "names": [name for name, _ in bucket],
            "dtypes": [str(tensor.dtype).removeprefix("torch.") for _, tensor in bucket],
            "shapes": [list(tensor.shape) for _, tensor in bucket],
            "group_name": GROUP_NAME.format(replica=replica),
            "flush_cache": False,
        }
        received = asyncio.create_task(self._post(replica, "/update_weights_from_distributed", body))

        def broadcast():
            for _, tensor in bucket:
                torch.distributed.broadcast(tensor, src=0, group=self._group)
            if self.backend == "nccl":
                torch.cuda.synchronize()

        await asyncio.to_thread(broadcast)
        await received

    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int | None = None,
        wire_format: str = "named_tensors",
        **kwargs: Any,
    ):
        if wire_format != "named_tensors":
            raise NotImplementedError(f"TokenSpeed rollout does not support wire_format={wire_format!r}")
        if kwargs.get("peft_config") is not None:
            raise NotImplementedError("TokenSpeed rollout does not support LoRA adapters")
        # The receiving replica must have resumed its weights, which its own
        # leader does just before this call.
        if torch.distributed.is_initialized():
            await asyncio.to_thread(torch.distributed.barrier)

        replica = self.target_replica
        if replica is not None:
            await self._ensure_group(replica)
        bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20
        bucket, size, count = [], 0, 0
        # Every rank iterates: gathering each full tensor is collective.
        async for name, tensor in ensure_async_iterator(weights):
            if replica is None:
                continue
            bucket.append((name, tensor.detach().contiguous()))
            size += tensor.numel() * tensor.element_size()
            count += 1
            if size >= bucket_bytes:
                await self._send_bucket(replica, bucket)
                bucket, size = [], 0
        if replica is None:
            return
        if bucket:
            await self._send_bucket(replica, bucket)
        async with self._session.get(await self._url(replica, "/flush_cache")) as resp:
            resp.raise_for_status()
        if global_steps is not None:
            await ray.get_actor(server_name(replica)).set_global_steps.remote(global_steps)
        logger.info("Sent %d tensors to TokenSpeed replica %d (step %s)", count, replica, global_steps)
