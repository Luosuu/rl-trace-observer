"""Broadcast weights from one trainer rank to TokenSpeed servers.

TokenSpeed receives weights only over its own RL weight-sync API: the sender
creates a ``torch.distributed`` group as rank 0, each server joins it with its
TP ranks (``/init_weights_update_group``), and every bucket of full tensors is
broadcast while the servers receive it (``/update_weights_from_distributed``).
One :class:`WeightSender` serves any number of servers through one group, so a
bucket is broadcast once whatever the number of replicas.
"""

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from datetime import timedelta

import aiohttp
import ray
import torch

from .server import _free_port, server_name
from .weight_group import standalone

SYNC_TIMEOUT = timedelta(seconds=float(os.environ.get("RL_TRACE_TOKENSPEED_SYNC_TIMEOUT", "600")))


def _join_group(master_address: str, master_port: int, world_size: int, group_name: str, backend: str):
    """Create the weight-update group as its rank 0, the way TokenSpeed's workers join it."""
    from torch.distributed import distributed_c10d as c10d
    from torch.distributed.distributed_c10d import Backend, PrefixStore, _new_process_group_helper, _world, rendezvous

    store, rank, world_size = next(
        rendezvous(f"tcp://{master_address}:{master_port}", 0, world_size, timeout=SYNC_TIMEOUT)
    )
    store.set_timeout(SYNC_TIMEOUT)
    # Not a split of the trainer's own world (see .weight_group).
    with standalone(c10d):
        group, _ = _new_process_group_helper(
            world_size,
            rank,
            [],
            Backend(backend),
            PrefixStore(group_name, store),
            group_name=group_name,
            backend_options=None,
            timeout=SYNC_TIMEOUT,
        )
    _world.pg_group_ranks[group] = {i: i for i in range(world_size)}
    return group


class WeightSender:
    """Rank 0 of a weight-update group whose other ranks are TokenSpeed servers.

    Args:
        replicas: replica ranks of the receiving servers, found by their actor names.
        group_name: the group's name; it must start with ``weight_group.GROUP_PREFIX``.
        backend: ``nccl``, or ``gloo`` for CPU tests.
        device: the sender's CUDA device; CUDA's current device is per thread,
            and the group is created and used from worker threads.
    """

    def __init__(self, replicas: Iterable[int], group_name: str, backend: str, device: int | None):
        self.replicas = list(replicas)
        self.group_name = group_name
        self.backend = backend
        self.device = device
        self._urls: dict[int, str] = {}
        self._session: aiohttp.ClientSession | None = None
        self._group = None

    async def url(self, replica: int, path: str) -> str:
        if replica not in self._urls:
            address, port = await ray.get_actor(server_name(replica)).get_server_address.remote()
            self._urls[replica] = f"http://{address}:{port}"
        return self._urls[replica] + path

    async def post(self, replica: int, path: str, body: dict | None = None) -> dict:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
                # The control server drops idle keep-alive connections; a reused one fails a later POST.
                connector=aiohttp.TCPConnector(force_close=True),
            )
        async with self._session.post(await self.url(replica, path), json=body or {}) as resp:
            payload = await resp.json(content_type=None)
            if resp.status != 200 or (isinstance(payload, dict) and payload.get("success") is False):
                raise RuntimeError(f"TokenSpeed replica {replica} {path} failed ({resp.status}): {payload}")
            return payload

    def _on_device(self, function, *args):
        if self.device is not None:
            torch.cuda.set_device(self.device)
        return function(*args)

    async def _ensure_group(self) -> None:
        if self._group is not None:
            return
        sizes = await asyncio.gather(*[ray.get_actor(server_name(r)).get_world_size.remote() for r in self.replicas])
        address = ray.util.get_node_ip_address().strip("[]")
        port = _free_port()
        world_size = 1 + sum(sizes)
        offsets = [1 + sum(sizes[:index]) for index in range(len(sizes))]
        joined = [
            asyncio.create_task(
                self.post(
                    replica,
                    "/init_weights_update_group",
                    {
                        "master_address": address,
                        "master_port": port,
                        "rank_offset": offset,
                        "world_size": world_size,
                        "group_name": self.group_name,
                        "backend": self.backend,
                    },
                )
            )
            for replica, offset in zip(self.replicas, offsets, strict=True)
        ]
        self._group = await asyncio.to_thread(
            self._on_device, _join_group, address, port, world_size, self.group_name, self.backend
        )
        await asyncio.gather(*joined)
        print(f"[rl-trace-observer] sending weights to TokenSpeed replicas {self.replicas} ({self.backend})")

    async def _send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        body = {
            "names": [name for name, _ in bucket],
            "dtypes": [str(tensor.dtype).removeprefix("torch.") for _, tensor in bucket],
            "shapes": [list(tensor.shape) for _, tensor in bucket],
            "group_name": self.group_name,
            "flush_cache": False,
        }
        received = [
            asyncio.create_task(self.post(replica, "/update_weights_from_distributed", body))
            for replica in self.replicas
        ]

        def broadcast():
            for _, tensor in bucket:
                torch.distributed.broadcast(tensor, src=0, group=self._group)
            if self.backend == "nccl":
                torch.cuda.synchronize()

        await asyncio.to_thread(self._on_device, broadcast)
        await asyncio.gather(*received)

    async def send(self, weights: AsyncIterator[tuple[str, torch.Tensor]], bucket_bytes: int) -> int:
        """Broadcast ``weights`` in buckets of about ``bucket_bytes``; returns the number of tensors."""
        await self._ensure_group()
        bucket, size, count = [], 0, 0
        async for name, tensor in weights:
            bucket.append((name, tensor.detach().contiguous()))
            size += tensor.numel() * tensor.element_size()
            count += 1
            if size >= bucket_bytes:
                await self._send_bucket(bucket)
                bucket, size = [], 0
        if bucket:
            await self._send_bucket(bucket)
        return count

    async def finish(self, global_steps: int | None) -> None:
        """Drop cached prefixes computed with the old weights and record the new version."""
        for replica in self.replicas:
            async with self._session.get(await self.url(replica, "/flush_cache")) as resp:
                resp.raise_for_status()
            if global_steps is not None:
                await ray.get_actor(server_name(replica)).set_global_steps.remote(global_steps)
