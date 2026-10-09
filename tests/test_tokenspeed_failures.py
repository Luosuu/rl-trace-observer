"""TokenSpeed rollout failure paths: misconfiguration, missing profiles, rejected weight updates, failed startup."""

import asyncio
import subprocess
import sys
import time
import types

import pytest
from verl.workers.rollout.replica import RolloutMode

from rl_trace_observer.integrations.tokenspeed import server as server_mod
from rl_trace_observer.integrations.tokenspeed import weight_sync
from rl_trace_observer.integrations.tokenspeed.replica import _check_weight_sync

ROLLOUT_DIR = "rollout/replica0"


@pytest.mark.parametrize(
    ("mode", "backend", "error"),
    [
        (RolloutMode.HYBRID, "naive", None),
        (RolloutMode.HYBRID, "tokenspeed", "hybrid TokenSpeed replicas"),
        (RolloutMode.STANDALONE, "tokenspeed", None),
        (RolloutMode.STANDALONE, "nccl", "Standalone TokenSpeed replicas"),
    ],
)
def test_a_backend_that_would_leave_replicas_without_weights_is_rejected(mode, backend, error):
    if error is None:
        _check_weight_sync(mode, backend)
    else:
        with pytest.raises(ValueError, match=error):
            _check_weight_sync(mode, backend)


def test_reward_and_teacher_servers_do_not_take_the_policy_names():
    names = {
        server_mod.server_name(0),
        server_mod.server_name(0, server_mod.REWARD),
        server_mod.server_name(0, server_mod.TEACHER, "_judge"),
    }
    assert len(names) == 3 and all(name.startswith(server_mod.SERVER_NAME_PREFIX) for name in names)


def _server(tmp_path, monkeypatch, process=None):
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv(server_mod.ACTIVITIES_ENV, "VIZTRACER,PROTON")
    server = server_mod.TokenSpeedServer.__new__(server_mod.TokenSpeedServer)
    server.replica_rank, server.world_size, server.output_component = 0, 1, server_mod.ROLLOUT
    server._profile_lock, server._registrations, server._profile = asyncio.Lock(), set(), None
    server._process = process or subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    server._session = None
    return server


def _profile(server, tmp_path):
    output_dir = tmp_path / ROLLOUT_DIR / "p"
    output_dir.mkdir(parents=True)
    server._profile = ("p", 3, output_dir)
    return output_dir


async def _stop_and_wait(server):
    start = time.monotonic()
    await server.stop_profile()
    await server.wait_for_profiles()
    return time.monotonic() - start


def _registered(tmp_path):
    from rl_trace_observer.process_record import record_path

    record = record_path(tmp_path)
    return record.read_text() if record.exists() else ""


def test_registration_stops_waiting_for_a_file_the_server_failed_to_write(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    output_dir = _profile(server, tmp_path)

    async def post(path, body=None, timeout=None):
        (output_dir / "t-TP0.viztracer.json").write_text("{}")
        (output_dir / "t-TP0.proton.chrome_trace.failed").write_text("Traceback ...")
        return {}

    server._post = post
    try:
        assert asyncio.run(_stop_and_wait(server)) < 10
    finally:
        server._process.kill()
    assert "t-TP0.viztracer.json" in _registered(tmp_path)


def test_registration_stops_waiting_when_the_server_exits(tmp_path, monkeypatch):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    server = _server(tmp_path, monkeypatch, process=dead)
    _profile(server, tmp_path)

    async def post(path, body=None, timeout=None):
        raise RuntimeError("connection refused")

    server._post = post
    assert asyncio.run(_stop_and_wait(server)) < 5


def test_registration_gives_up_once_files_stop_appearing(tmp_path, monkeypatch):
    monkeypatch.setenv(server_mod.STALL_TIMEOUT_ENV, "2")
    server = _server(tmp_path, monkeypatch)
    _profile(server, tmp_path)

    async def post(path, body=None, timeout=None):
        return {}

    server._post = post
    try:
        assert asyncio.run(_stop_and_wait(server)) < 10
    finally:
        server._process.kill()


def test_a_failed_start_is_stopped_so_later_profiles_can_start(tmp_path, monkeypatch):
    server = _server(tmp_path, monkeypatch)
    posts = []

    async def post(path, body=None, timeout=None):
        posts.append(path)
        if path == "/start_profile":
            raise TimeoutError("waited 300 s for the previous profile")
        return {}

    server._post = post
    try:
        asyncio.run(server.start_profile(global_step=3))
    finally:
        server._process.kill()
    assert posts == ["/start_profile", "/stop_profile"]
    assert server._profile is None


def test_a_server_that_never_becomes_ready_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv(server_mod.STARTUP_TIMEOUT_ENV, "1")
    monkeypatch.setenv(server_mod.ACTIVITIES_ENV, "")
    server = _server(tmp_path, monkeypatch)
    server._process.kill()
    server._process = None
    server._address, server._control_port, server.cuda_visible_devices = "127.0.0.1", 9, ""
    monkeypatch.setattr(server, "_command", lambda: [sys.executable, "-c", "import time; time.sleep(60)"])

    with pytest.raises(TimeoutError):
        asyncio.run(server.launch_server())
    assert server._process.poll() is not None, "tokenspeed serve must not keep its GPUs"


class _Remote:
    def __init__(self, value):
        self.value = value

    def remote(self):
        async def get():
            return self.value

        return get()


def test_a_rejected_weight_update_group_raises_the_server_error_at_once(monkeypatch):
    actor = types.SimpleNamespace(get_world_size=_Remote(2), get_server_address=_Remote(("127.0.0.1", 1)))
    monkeypatch.setattr(
        weight_sync,
        "ray",
        types.SimpleNamespace(
            get_actor=lambda name: actor, util=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
        ),
    )
    # No server joins, so the rendezvous would wait until its timeout.
    monkeypatch.setattr(weight_sync, "_join_group", lambda *args: time.sleep(5))
    sender = weight_sync.WeightSender([0], "group", "gloo", None)

    async def post(replica, path, body=None):
        raise RuntimeError(f"replica {replica} {path} failed: unsupported backend")

    sender.post = post

    async def ensure_group():
        start = time.monotonic()
        with pytest.raises(RuntimeError, match="unsupported backend"):
            await sender._ensure_group()
        return time.monotonic() - start

    # (asyncio.run itself then waits for the blocked thread; a trainer's loop keeps running.)
    assert asyncio.run(ensure_group()) < 2
