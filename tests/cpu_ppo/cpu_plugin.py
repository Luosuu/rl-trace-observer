"""Test-only VERL plugin that runs PPO on CPU with a mock rollout server.

Loaded in every VERL process through ``VERL_USE_EXTERNAL_MODULES`` together
with ``VERL_PLATFORM=cpu``. It only uses VERL's extension points:

* ``PlatformRegistry``: a ``cpu`` platform with the ``gloo`` backend.
* ``EngineRegistry``: the FSDP engines, registered for ``device="cpu"``.
* ``_ROLLOUT_REGISTRY`` / ``RolloutReplicaRegistry``: a ``mock`` rollout whose
  server returns random token sequences at a fixed pace and records one
  ``mock_generate`` span per request, like VERL's vLLM and SGLang servers do
  per replica.

VERL 0.9.1 gaps for a cpu platform, worked around here or in the test config:

* ``initialize_global_process_group_ray`` builds the backend
  ``"cpu:gloo,{device}:{backend}"``, which is invalid when the device is cpu.
* Many call sites pass ``get_device_id()`` to ``.to()``/``device=``, where an
  integer means a CUDA index, so the platform returns ``torch.device("cpu")``.
* ``ResourcePoolManager`` only counts Ray ``GPU``/``NPU`` resources, so the run
  declares logical GPUs.
* FSDP1 passes an integer ``device_id`` to ``FullyShardedDataParallel``; the
  test uses FSDP2.
* The FSDP engines are only registered for cuda and npu.
"""

import asyncio
import logging
import math
import os
import random
from contextlib import contextmanager
from types import ModuleType
from typing import Any

import ray
import torch
from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry

logger = logging.getLogger(__name__)


class _CpuDeviceModule(ModuleType):
    """Stands in for ``torch.cuda``: ``torch.cpu`` where it exists, otherwise a no-op."""

    def __init__(self):
        super().__init__("rl_trace_observer_cpu_device")

    @staticmethod
    def mem_get_info(device=None) -> tuple[int, int]:
        page = os.sysconf("SC_PAGE_SIZE")
        return os.sysconf("SC_AVPHYS_PAGES") * page, os.sysconf("SC_PHYS_PAGES") * page

    @staticmethod
    def get_device_name(device=None) -> str:
        return "CPU"

    @staticmethod
    def get_rng_state(device=None) -> torch.Tensor:
        return torch.get_rng_state()

    @staticmethod
    def set_rng_state(state: torch.Tensor, device=None) -> None:
        torch.set_rng_state(state)

    @staticmethod
    def manual_seed(seed: int) -> None:
        torch.manual_seed(seed)

    manual_seed_all = manual_seed

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if hasattr(torch.cpu, name):
            return getattr(torch.cpu, name)
        logger.debug("CPU device module: no-op for %s", name)
        return lambda *args, **kwargs: 0


@PlatformRegistry.register(platform="cpu")
class PlatformCPU(PlatformBase):
    _device_module = _CpuDeviceModule()

    @property
    def device_name(self) -> str:
        return "cpu"

    @property
    def vendor_name(self) -> str:
        return "cpu"

    @property
    def device_module(self) -> ModuleType:
        return self._device_module

    def is_available(self) -> bool:
        return True

    def is_platform_available(self, use_smi_check=False) -> bool:
        # Only selected explicitly through VERL_PLATFORM=cpu.
        return False

    def current_device(self) -> torch.device:
        # VERL passes this to .to()/device=; an integer index would mean cuda:<index>.
        return torch.device("cpu")

    def device_count(self) -> int:
        return 1

    def set_device(self, device_index: int) -> None:
        pass

    def synchronize(self, device_index=None) -> None:
        pass

    def manual_seed(self, seed: int) -> None:
        torch.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)

    def set_allocator_settings(self, settings: str) -> None:
        pass

    def empty_cache(self) -> None:
        pass

    def get_device_capability(self, device_index: int = 0):
        return None, None

    def communication_backend_name(self) -> str:
        return "gloo"

    def visible_devices_envvar(self) -> str:
        return "RL_TRACE_OBSERVER_VISIBLE_CPUS"

    @contextmanager
    def nvtx_range(self, msg: str):
        yield

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def ray_resource_name(self) -> str:
        # VERL's resource manager counts Ray "GPU" resources, so the CPU run
        # declares logical GPUs (ray_init.num_gpus) that only drive scheduling.
        return "GPU"

    def ray_noset_envvars(self) -> list[str]:
        return []

    def is_ipc_supported(self) -> bool:
        return False

    def cudart(self) -> Any:
        return None


# VERL modules below query the platform at import time, so they must be imported
# only after the cpu platform is registered.
from verl.utils import distributed as verl_distributed  # noqa: E402
from verl.utils.tracking import RLInsightLogger  # noqa: E402
from verl.workers.engine.base import EngineRegistry  # noqa: E402
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead, FSDPEngineWithValueHead  # noqa: E402
from verl.workers.rollout import base as rollout_base  # noqa: E402
from verl.workers.rollout.base import BaseRollout  # noqa: E402
from verl.workers.rollout.replica import RolloutReplica, RolloutReplicaRegistry, TokenOutput  # noqa: E402

# VERL gap: the default backend "cpu:gloo,{device}:{backend}" becomes the
# invalid "cpu:gloo,cpu:gloo" on a cpu platform. Patched before engine_workers
# imports the function by name.
_initialize_global_process_group_ray = verl_distributed.initialize_global_process_group_ray


def _initialize_cpu_process_group(timeout_second=None, backend=None):
    return _initialize_global_process_group_ray(timeout_second=timeout_second, backend=backend or "gloo")


verl_distributed.initialize_global_process_group_ray = _initialize_cpu_process_group

EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="cpu")(FSDPEngineWithLMHead)
EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="cpu")(FSDPEngineWithValueHead)


def _server_name(replica_rank: int) -> str:
    return f"mock_server_{replica_rank}"


class MockServerAdapter(BaseRollout):
    """Rollout side of the training worker: discards weights, forwards the step.

    Like VERL's vLLM adapter, the first rank of each replica tells the server
    which trainer step its weights come from; the trainer reads it back from
    every rollout output.
    """

    def __init__(self, config, model_config, device_mesh, *args, replica_rank: int = -1, **kwargs):
        super().__init__(config, model_config, device_mesh)
        rank = int(os.environ.get("RANK", "0"))
        world_size = (
            self.config.tensor_model_parallel_size
            * self.config.data_parallel_size
            * self.config.pipeline_model_parallel_size
        )
        self._replica_rank = rank // world_size if replica_rank == -1 else replica_rank
        self._owns_server = rank % world_size == 0

    async def resume(self, tags: list[str]):
        pass

    async def update_weights(self, weights, wire_format: str = "named_tensors", global_steps=None, **kwargs):
        for _ in weights:
            pass
        if self._owns_server and global_steps is not None:
            await ray.get_actor(_server_name(self._replica_rank)).set_global_steps.remote(global_steps)

    async def release(self):
        pass


@ray.remote(num_cpus=0)
class MockLLMServer:
    """Token-in, token-out server returning random completions at a fixed pace.

    Each request occupies a batch slot for its whole generation, like a request
    in an inference engine's running batch. The ``mock_generate`` span is
    recorded on the slot's lane (``replica_<r>/slot_<k>``) with the request id,
    so concurrent requests show up as separate, non-overlapping spans.
    """

    PREFILL_SECONDS = 0.05
    SECONDS_PER_TOKEN = 0.03

    def __init__(self, replica_rank: int, vocab_size: int, eos_token_id: int | None):
        self._replica_rank = replica_rank
        self._vocab_size = vocab_size
        self._eos_token_id = eos_token_id
        self._global_steps = None
        self._busy_slots: set[int] = set()

    async def set_global_steps(self, global_steps: int):
        self._global_steps = global_steps

    async def generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict, **kwargs) -> TokenOutput:
        slot = next(index for index in range(len(self._busy_slots) + 1) if index not in self._busy_slots)
        self._busy_slots.add(slot)
        try:
            lane = f"replica_{self._replica_rank}/slot_{slot}"
            with RLInsightLogger.trace_state("mock_generate", state_lane_id=lane, request_id=request_id):
                return await self._generate(request_id, prompt_ids, sampling_params)
        finally:
            self._busy_slots.discard(slot)

    async def _generate(self, request_id: str, prompt_ids: list[int], sampling_params: dict) -> TokenOutput:
        max_tokens = int(sampling_params.get("max_tokens") or sampling_params.get("max_new_tokens") or 8)
        rng = random.Random(f"{request_id}:{len(prompt_ids)}")
        length = rng.randint(1, max_tokens)
        token_ids = [rng.randrange(8, self._vocab_size) for _ in range(length)]
        if self._eos_token_id is not None and length < max_tokens:
            token_ids.append(self._eos_token_id)
        await asyncio.sleep(self.PREFILL_SECONDS + self.SECONDS_PER_TOKEN * len(token_ids))
        # Uniform sampling: every token has probability 1 / vocab_size.
        log_probs = [-math.log(self._vocab_size)] * len(token_ids)
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason="completed",
            extra_fields={"global_steps": self._global_steps},
        )

    async def wake_up(self):
        pass

    async def sleep(self):
        pass

    async def abort_all_requests(self, reject_request: bool = False):
        pass

    async def resume_generation(self):
        pass

    async def clear_kv_cache(self):
        pass

    async def release_kv_cache(self):
        pass

    async def resume_kv_cache(self):
        pass

    async def start_profile(self, **kwargs):
        pass

    async def stop_profile(self):
        pass


class MockReplica(RolloutReplica):
    async def launch_servers(self):
        tokenizer = self.model_config.tokenizer if hasattr(self.model_config, "tokenizer") else None
        vocab_size = len(tokenizer) if tokenizer is not None else 64
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        server = MockLLMServer.options(name=_server_name(self.replica_rank) + self.name_suffix).remote(
            self.replica_rank, vocab_size, eos_token_id
        )
        self.servers = [server]
        self._server_handle = server
        self._server_address = f"127.0.0.1:{18000 + self.replica_rank}"


rollout_base._ROLLOUT_REGISTRY[("mock", "async")] = f"{__name__}.MockServerAdapter"
RolloutReplicaRegistry.register("mock", lambda: MockReplica)
