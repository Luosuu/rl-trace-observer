"""The Ray actor that runs one TokenSpeed server and speaks its HTTP API for VERL.

``TokenSpeedServer`` launches ``tokenspeed serve`` as a subprocess on the GPUs
of its replica and implements the server-actor interface VERL's agent loop and
replicas call (``generate``, ``sleep``, ``start_profile``, ...). Everything goes
through TokenSpeed's control port, which serves SGLang-style ``/generate`` and
the RL control routes. It needs our TokenSpeed fork (``Luosuu/tokenspeed``,
branch ``tianle/rl-trace``) for Proton under CUDA graphs, background profile
writes and weight-update groups under torch 2.14.

Profiling: each profiled step asks TokenSpeed for ``VIZTRACER`` and ``PROTON``
traces (``RL_TRACE_TOKENSPEED_PROFILE_ACTIVITIES``) in a directory of its own,
``<RL_TRACE_OUTPUT_DIR>/rollout/replica<r>/<run_id>-step-<n>``. The directory,
not the file name, identifies the profile: ``tokenspeed serve`` names the files
after a timestamp whatever ``profile_id`` is sent. Proton profiles come from
one session per scheduler that sees the CUDA-graph captures
(``TOKENSPEED_PROTON_SESSION_DIR``). The scheduler processes do not register
artifacts, so once ``/stop_profile`` returns this actor registers
each rank's files in its own process record, with the step and the scheduler's
``rank_tag``.
"""

import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import ray

from rl_trace_observer.context import (
    REQUEST_FLOW_ATTRIBUTE,
    REQUEST_ID_ATTRIBUTE,
    current_run_id,
    profile_session_id,
)
from rl_trace_observer.merger.sources import TOKENSPEED_PROTON, TOKENSPEED_VIZTRACER, tokenspeed_profile_file

logger = logging.getLogger(__name__)

COMMAND_ENV = "RL_TRACE_TOKENSPEED_COMMAND"
EXTRA_ARGS_ENV = "RL_TRACE_TOKENSPEED_ARGS"
ACTIVITIES_ENV = "RL_TRACE_TOKENSPEED_PROFILE_ACTIVITIES"
STARTUP_TIMEOUT_ENV = "RL_TRACE_TOKENSPEED_STARTUP_TIMEOUT"
DEFAULT_ACTIVITIES = "VIZTRACER,PROTON"
TOKENSPEED_ENV_DEFAULTS = {
    # Proton writes a mergeable timeline only as a Chrome trace of trace-mode data.
    "TOKENSPEED_KERNEL_PROFILE_DATA": "trace",
    "TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT": "chrome_trace",
}
# Each scheduler keeps one Proton session from before its CUDA-graph captures.
PROTON_SESSION_ENV = "TOKENSPEED_PROTON_SESSION_DIR"
_ACTIVITY_KINDS = {"VIZTRACER": TOKENSPEED_VIZTRACER, "PROTON": TOKENSPEED_PROTON}
# Our TokenSpeed fork's ranks write their files after /stop_profile returns;
# registration waits for them in the background. It gives up once no file has
# appeared or grown for this long (seconds).
STALL_TIMEOUT_ENV = "RL_TRACE_TOKENSPEED_PROFILE_STALL_TIMEOUT"
# A file the fork failed to write leaves this marker next to it instead.
FAILED_SUFFIX = ".failed"


SERVER_NAME_PREFIX = "tokenspeed_server_"
# What a replica serves: the policy's rollout, or a frozen reward or teacher model.
ROLLOUT, REWARD, TEACHER = "rollout", "reward", "teacher"


def server_name(replica_rank: int, role: str = ROLLOUT, name_suffix: str = "") -> str:
    """The Ray actor name of a replica's server; weights are only ever sent to ``ROLLOUT`` servers."""
    return f"{SERVER_NAME_PREFIX}{role}_{replica_rank}{name_suffix}"


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _die_with_parent() -> None:
    """Have Linux stop the server when this actor's process exits, e.g. when Ray kills it."""
    if sys.platform == "linux":
        import ctypes
        import signal

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _complete_files(
    output_dir: Path, kinds: set[str], sizes: dict[Path, int]
) -> tuple[dict[Path, tuple[str, str, str]], list[Path]]:
    """Profile files of ``kinds`` that are complete, and the outputs the server failed to write.

    The fork renames VizTracer and Proton files into place once written; a
    file still counts as complete only if it holds a whole JSON object and
    stopped growing since ``sizes``, for writers that write in place. Cheap
    enough for files of hundreds of MB: it reads the last bytes only. ``sizes``
    is updated for the next call.
    """
    files, failed = {}, []
    for path in sorted(output_dir.iterdir()) if output_dir.is_dir() else []:
        if path.name.endswith(FAILED_SUFFIX):
            parsed = tokenspeed_profile_file(path.with_name(path.name.removesuffix(FAILED_SUFFIX)))
            if parsed and parsed[0] in kinds:
                failed.append(path)
            continue
        parsed = tokenspeed_profile_file(path)
        if not parsed or parsed[0] not in kinds:
            continue
        try:
            size = path.stat().st_size
            with path.open("rb") as file:
                file.seek(max(0, size - 16))
                ends_object = file.read().rstrip().endswith(b"}")
        except OSError:
            continue
        if size > 0 and ends_object and sizes.get(path) == size:
            files[path] = parsed
        sizes[path] = size
    return files, failed


class TokenSpeedServer:
    """One TokenSpeed server (all TP ranks of one replica), driven over HTTP."""

    def __init__(
        self,
        config: Any,
        model_config: Any,
        replica_rank: int,
        cuda_visible_devices: str,
        free_cache_engine: bool,
        role: str,
        name_suffix: str,
    ):
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig

        self.config = omega_conf_to_dataclass(config)
        self.model_config = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.replica_rank = replica_rank
        # Where this replica's profiles go and the role its artifacts are registered under.
        self.output_component = ROLLOUT if role == ROLLOUT and not name_suffix else f"{role}{name_suffix}"
        self.cuda_visible_devices = cuda_visible_devices
        self.free_cache_engine = free_cache_engine
        self.world_size = (
            config.tensor_model_parallel_size * config.data_parallel_size * config.pipeline_model_parallel_size
        )
        # Model weights version, set by the adapter after each weight update.
        self.global_steps = None
        self._address = ray.util.get_node_ip_address().strip("[]")
        self._port = _free_port()
        self._control_port = _free_port()
        self._process: subprocess.Popen | None = None
        self._session: aiohttp.ClientSession | None = None
        self._busy_slots: set[int] = set()
        # (profile_id, global_step, output_dir) of the profile in progress.
        self._profile: tuple[str, int | None, Path] | None = None
        # Keeps /start_profile and /stop_profile in order when callers do not wait.
        self._profile_lock = asyncio.Lock()
        # Registrations of stopped profiles whose files are still being written.
        self._registrations: set[asyncio.Task] = set()

    # ------------------------------------------------------------------ #
    # Process lifecycle
    # ------------------------------------------------------------------ #

    def _command(self) -> list[str]:
        command = shlex.split(os.environ.get(COMMAND_ENV, "")) or [sys.executable, "-m", "tokenspeed.cli", "serve"]
        config = self.config
        args = [
            "--model", str(self.model_config.local_path),
            "--tp", str(config.tensor_model_parallel_size),
            "--host", "0.0.0.0",
            "--port", str(self._port),
            "--control-port", str(self._control_port),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--dtype", str(config.dtype),
            "--max-num-seqs", str(config.max_num_seqs),
            "--enable-output-logprobs",
        ]  # fmt: skip
        if config.max_model_len:
            args += ["--max-model-len", str(config.max_model_len)]
        if self.free_cache_engine:
            args.append("--enable-memory-saver")
        if config.enforce_eager:
            args.append("--enforce-eager")
        if self.model_config.trust_remote_code:
            args.append("--trust-remote-code")
        return command + args + shlex.split(os.environ.get(EXTRA_ARGS_ENV, ""))

    async def launch_server(self) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices}
        for key, value in TOKENSPEED_ENV_DEFAULTS.items():
            env.setdefault(key, value)
        if "PROTON" in self._activities():
            # One Proton session per scheduler from startup, which sees the CUDA-graph captures.
            from rl_trace_observer.output import trace_output_dir

            try:
                session_dir = self._replica_dir(trace_output_dir()) / "proton-session"
            except ValueError as error:
                logger.warning("TokenSpeed Proton profiles need --enforce-eager: %s", error)
            else:
                env.setdefault(PROTON_SESSION_ENV, str(session_dir))
        command = self._command()
        logger.info("Replica %d: launching %s", self.replica_rank, shlex.join(command))
        # Output goes to this actor's Ray log.
        self._process = subprocess.Popen(
            command, env=env, stdout=sys.stdout, stderr=sys.stderr, preexec_fn=_die_with_parent
        )
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            # The control server drops idle keep-alive connections; a reused one fails a later POST.
            connector=aiohttp.TCPConnector(force_close=True),
        )

        try:
            await self._wait_until_ready(float(os.environ.get(STARTUP_TIMEOUT_ENV, "1800")))
        except BaseException:
            # Ray keeps the actor alive after a failed call, so the server would keep its GPUs.
            await self._stop_process()
            raise
        logger.info("Replica %d: TokenSpeed ready at %s", self.replica_rank, self._url(""))
        self._set_process_role()

    def _replica_dir(self, root: Path) -> Path:
        return root / self.output_component / f"replica{self.replica_rank}"

    def _set_process_role(self) -> None:
        try:
            from rl_trace_observer.output import trace_output_dir
            from rl_trace_observer.process_record import set_process_role

            set_process_role(trace_output_dir(), "rollout_server")
        except Exception:
            logger.debug("No process record for the TokenSpeed server actor", exc_info=True)

    async def _wait_until_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(f"tokenspeed serve exited with {self._process.returncode} before becoming ready")
            try:
                async with self._session.get(self._url("/health"), timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        return
            except (TimeoutError, aiohttp.ClientError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"tokenspeed serve not ready on {self._url('/health')}")
            await asyncio.sleep(2)

    async def _stop_process(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            await asyncio.to_thread(self._process.wait, 30)
        except subprocess.TimeoutExpired:
            self._process.kill()

    def shutdown(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()

    def get_world_size(self) -> int:
        """The number of TokenSpeed ranks, which all join a weight-update group."""
        return self.world_size

    def get_server_address(self) -> tuple[str, int]:
        """The control port: ``/generate`` and the RL control routes."""
        return self._address, self._control_port

    def _url(self, path: str) -> str:
        return f"http://{self._address}:{self._control_port}{path}"

    async def _post(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        assert self._session is not None, "server not launched"
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with self._session.post(self._url(path), json=body or {}, timeout=client_timeout) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise RuntimeError(f"POST {path} failed with {resp.status}: {text[:2000]}")
            return json.loads(text) if text else {}

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Any = None,
        video_data: Any = None,
        **kwargs: Any,
    ):
        from verl.utils.tracking import RLInsightLogger
        from verl.workers.rollout.replica import TokenOutput

        if image_data or video_data:
            raise NotImplementedError("TokenSpeed rollout does not support multi-modal inputs yet")
        config = self.config
        prompt_ids = list(prompt_ids)
        sampling_params = dict(sampling_params)
        if "max_new_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_new_tokens")
        elif "max_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_tokens")
        else:
            max_new_tokens = min(
                config.response_length, config.prompt_length + config.response_length - len(prompt_ids)
            )
        if config.max_model_len:
            max_new_tokens = min(max_new_tokens, config.max_model_len - len(prompt_ids) - 1)
        sampling_params["max_new_tokens"] = max(0, int(max_new_tokens))
        return_logprob = bool(sampling_params.pop("logprobs", False))
        body = {
            "rid": request_id,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
        }

        # Concurrent requests get their own lanes, so their spans never overlap.
        slot = next(index for index in range(len(self._busy_slots) + 1) if index not in self._busy_slots)
        self._busy_slots.add(slot)
        try:
            lane = f"replica_{self.replica_rank}/slot_{slot}"
            # A point on the request's path, after the agent loop's span for it (see verl.requests).
            labels = {REQUEST_ID_ATTRIBUTE: request_id, REQUEST_FLOW_ATTRIBUTE: True}
            with RLInsightLogger.trace_state("tokenspeed_generate", state_lane_id=lane, **labels):
                output = await self._post("/generate", body)
        finally:
            self._busy_slots.discard(slot)

        if isinstance(output, list):
            output = output[0]
        meta_info = output.get("meta_info") or {}
        finish_reason = meta_info.get("finish_reason")
        stop_reason = finish_reason.get("type") if isinstance(finish_reason, dict) else finish_reason
        token_ids = list(output.get("output_ids") or [])
        log_probs = None
        if return_logprob:
            entries = meta_info.get("output_token_logprobs") or []
            if not token_ids:
                token_ids = [int(entry[1]) for entry in entries]
            if len(entries) == len(token_ids):
                log_probs = [float(entry[0]) for entry in entries]
            else:
                logger.error(
                    "Request %s: %d logprobs for %d tokens; dropping the response",
                    request_id,
                    len(entries),
                    len(token_ids),
                )
                token_ids, log_probs = [], []
        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=stop_reason,
            extra_fields={"global_steps": self.global_steps},
        )

    async def set_global_steps(self, global_steps: int) -> None:
        self.global_steps = global_steps

    # ------------------------------------------------------------------ #
    # Memory and request control
    # ------------------------------------------------------------------ #

    async def sleep(self) -> None:
        # The trainer stops rollout profiling only after sleeping the replicas;
        # stop it first so the profile does not cover releasing memory.
        await self.stop_profile()
        if self.free_cache_engine:
            await self._post("/release_memory_occupation", {"tags": ["kv_cache", "weights"]})

    async def wake_up(self) -> None:
        if self.free_cache_engine:
            await self._post("/resume_memory_occupation", {"tags": ["weights", "kv_cache"]})

    async def release_kv_cache(self) -> None:
        if self.free_cache_engine:
            await self._post("/release_memory_occupation", {"tags": ["kv_cache"]})

    async def resume_kv_cache(self) -> None:
        if self.free_cache_engine:
            await self._post("/resume_memory_occupation", {"tags": ["kv_cache"]})
        await self.clear_kv_cache()

    async def clear_kv_cache(self) -> None:
        assert self._session is not None, "server not launched"
        async with self._session.get(self._url("/flush_cache")) as resp:
            if resp.status != 200:
                raise RuntimeError(f"GET /flush_cache failed with {resp.status}: {await resp.text()}")

    async def abort_all_requests(self, reject_request: bool = False) -> None:
        # TokenSpeed resumes admission right after aborting, like SGLang.
        await self._post("/abort_request", {"abort_all": True})

    async def resume_generation(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Profiling
    # ------------------------------------------------------------------ #

    @staticmethod
    def _activities() -> list[str]:
        activities = os.environ.get(ACTIVITIES_ENV, DEFAULT_ACTIVITIES).upper().split(",")
        return [activity.strip() for activity in activities if activity.strip()]

    async def start_profile(self, global_step: int | None = None, **kwargs: Any) -> None:
        """Start a TokenSpeed profile for ``global_step``; a failure is logged, never raised."""
        async with self._profile_lock:
            await self._start_profile(global_step)

    async def _start_profile(self, global_step: int | None) -> None:
        activities = self._activities()
        if not activities or self._profile is not None:
            return
        run_id = current_run_id() or "run"
        profile_id = profile_session_id(run_id, global_step) or f"{run_id}-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            from rl_trace_observer.output import safe_component, trace_output_dir

            output_dir = self._replica_dir(trace_output_dir()) / safe_component(profile_id)
        except ValueError as error:
            logger.warning("Not profiling TokenSpeed: %s", error)
            return
        # TokenSpeed starts a profile only once the previous one is written; wait
        # here, where it is visible, rather than inside /start_profile.
        await asyncio.gather(*self._registrations)
        body = {"output_dir": str(output_dir), "activities": activities, "profile_id": profile_id}
        try:
            await self._post("/start_profile", body, timeout=300)
        except Exception as error:
            logger.warning("TokenSpeed /start_profile %s failed; continuing without it: %s", body, error)
            # A start that timed out may still begin later; stop it, or every later start is rejected.
            try:
                await self._post("/stop_profile", timeout=600)
            except Exception:
                pass
            return
        self._profile = (profile_id, global_step, output_dir)

    async def stop_profile(self) -> None:
        """Stop the profile in progress; each rank's files are registered once written, in the background."""
        async with self._profile_lock:
            if self._profile is None:
                return
            profile_id, global_step, output_dir = self._profile
            self._profile = None
            try:
                await self._post("/stop_profile", timeout=600)
            except Exception as error:
                # Whatever is written is still registered, and reported
                # missing or incomplete by the merger.
                logger.warning("TokenSpeed /stop_profile failed for %s: %s", profile_id, error)
            task = asyncio.create_task(self._register_profile_logged(profile_id, global_step, output_dir))
            self._registrations.add(task)
            task.add_done_callback(self._registrations.discard)

    async def wait_for_profiles(self) -> None:
        """Wait until every stopped profile's files are written and registered."""
        # A stop submitted earlier but still running creates its registration under the lock.
        async with self._profile_lock:
            registrations = list(self._registrations)
        await asyncio.gather(*registrations)

    async def _register_profile_logged(self, profile_id: str, global_step: int | None, output_dir: Path) -> None:
        try:
            await self._register_profile(profile_id, global_step, output_dir)
        except Exception:
            logger.exception("Failed to register the TokenSpeed profile %s", profile_id)

    async def _register_profile(self, profile_id: str, global_step: int | None, output_dir: Path) -> None:
        from rl_trace_observer.output import trace_output_dir
        from rl_trace_observer.process_record import register_artifact

        kinds = {_ACTIVITY_KINDS[a] for a in self._activities() if a in _ACTIVITY_KINDS}
        expected = len(kinds) * self.world_size
        stall_timeout = float(os.environ.get(STALL_TIMEOUT_ENV, "300"))
        sizes: dict[Path, int] = {}
        progress, last_progress = None, time.monotonic()
        while True:
            exited = self._process is not None and self._process.poll() is not None
            # Off the event loop, which keeps serving generation requests.
            files, failed = await asyncio.to_thread(_complete_files, output_dir, kinds, sizes)
            if len(files) + len(failed) >= expected or exited:
                break
            if (state := (len(files), dict(sizes))) != progress:
                progress, last_progress = state, time.monotonic()
            elif time.monotonic() - last_progress > stall_timeout:
                break
            await asyncio.sleep(1.0)
        if failed:
            logger.warning("TokenSpeed profile %s: the server failed to write %s", profile_id, [p.name for p in failed])
        if len(files) < expected:
            logger.warning(
                "TokenSpeed profile %s: found %d of %d files%s",
                profile_id,
                len(files),
                expected,
                " (the server exited)" if exited else "",
            )
        root = trace_output_dir()
        for path, (kind, _, rank_tag) in files.items():
            register_artifact(
                root,
                kind,
                path,
                global_step=global_step,
                role=f"{self.output_component}_replica{self.replica_rank}",
                rank_tag=rank_tag,
            )
