"""``rollout.name=tokenspeed``: a VERL rollout replica served by TokenSpeed.

In VERL's synchronous trainer the replica is hybrid: its TokenSpeed server runs
on the GPUs of ``world_size`` training workers, sleeps (releases weights and KV
cache) while they train, and receives new weights from them after every step
(see :mod:`.adapter`). One replica runs on one node.
"""

import asyncio
import os

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl.utils.device import get_visible_devices_keyword
from verl.workers.rollout.replica import RolloutMode, RolloutReplica

from .server import TokenSpeedServer, server_name


class TokenSpeedReplica(RolloutReplica):
    async def launch_servers(self):
        if self.nnodes != 1:
            raise NotImplementedError("A TokenSpeed replica must fit on one node")
        if self.rollout_mode != RolloutMode.HYBRID:
            raise NotImplementedError(f"TokenSpeed rollout supports hybrid mode only, got {self.rollout_mode}")
        assert len(self.workers) == self.world_size, f"{len(self.workers)} workers for world size {self.world_size}"

        keyword = get_visible_devices_keyword()
        infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self, keyword=keyword: (ray.get_runtime_context().get_node_id(), os.environ.get(keyword, ""))
                )
                for worker in self.workers
            ]
        )
        devices = sorted({int(device) for _, visible in infos for device in visible.split(",") if device.strip()})
        server = (
            ray.remote(TokenSpeedServer)
            .options(
                name=server_name(self.replica_rank, self.name_suffix),
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=infos[0][0], soft=False),
                max_concurrency=self.max_concurrency,
            )
            .remote(
                config=self.config,
                model_config=self.model_config,
                replica_rank=self.replica_rank,
                cuda_visible_devices=",".join(map(str, devices)),
                free_cache_engine=self.config.free_cache_engine,
            )
        )
        await server.launch_server.remote()
        address, port = await server.get_server_address.remote()
        self.servers = [server]
        self._server_handle = server
        self._server_address = f"{address}:{port}"
