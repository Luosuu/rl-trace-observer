"""The agent loop's rollout_request span: one per model call, with the id its server was sent."""

import asyncio
import sys
import types

import pytest

from rl_trace_observer.context import REQUEST_FLOW_ATTRIBUTE
from rl_trace_observer.integrations.verl import requests


@pytest.fixture
def spans(monkeypatch):
    recorded = []

    class Logger:
        @staticmethod
        def enabled():
            return True

        @staticmethod
        def trace_span(name, **kwargs):
            recorded.append((name, kwargs))

    monkeypatch.setitem(sys.modules, "verl.utils.tracking", types.SimpleNamespace(RLInsightLogger=Logger))
    return recorded


def _module():
    class LLMServerClient:
        """The shape of VERL's client: pick a server by the sticky id, send it a fresh one."""

        def __init__(self):
            self.sent = []

        async def _acquire_server(self, request_id, **kwargs):
            return f"server-for-{request_id}", object()

        def _vllm_request_id(self, request_id):
            return f"fresh-{len(self.sent)}-{request_id}"

        async def generate(self, request_id, *, prompt_ids, sampling_params):
            server_id, _ = await self._acquire_server(request_id)
            sent = self._vllm_request_id(request_id)
            self.sent.append((server_id, sent))
            await asyncio.sleep(0.01)
            return sent

    return types.SimpleNamespace(LLMServerClient=LLMServerClient)


def test_each_call_records_the_id_its_server_was_sent(spans):
    module = _module()
    assert requests.patch_server_client(module) is True
    assert requests.patch_server_client(module) is False
    manager = module.LLMServerClient()

    async def calls():
        return await asyncio.gather(
            manager.generate("traj-a", prompt_ids=[1], sampling_params={}),
            manager.generate("traj-b", prompt_ids=[2], sampling_params={}),
        )

    sent = asyncio.run(calls())

    recorded = {kwargs["attributes"]["request_id"]: (name, kwargs) for name, kwargs in spans}
    assert set(recorded) == set(sent)
    lanes = set()
    for (server_id, request_id), trajectory in zip(manager.sent, ("traj-a", "traj-b"), strict=True):
        name, kwargs = recorded[request_id]
        attributes = kwargs["attributes"]
        assert name == "rollout_request"
        assert attributes["trajectory_id"] == trajectory and attributes["server_id"] == server_id
        assert attributes[REQUEST_FLOW_ATTRIBUTE] is True
        assert kwargs["start_time_ns"] <= kwargs["end_time_ns"]
        lanes.add(attributes["state_lane_id"])
    assert len(lanes) == 2, "concurrent calls must not share a lane"


def test_a_failed_call_is_still_recorded(spans):
    module = _module()

    async def fail(self, request_id, **kwargs):
        self._vllm_request_id(request_id)
        raise RuntimeError("server gone")

    module.LLMServerClient.generate = fail
    requests.patch_server_client(module)

    with pytest.raises(RuntimeError):
        asyncio.run(module.LLMServerClient().generate("traj", prompt_ids=[], sampling_params={}))
    assert [attributes["attributes"]["trajectory_id"] for _, attributes in spans] == ["traj"]


def test_a_renamed_client_fails_clearly():
    with pytest.raises(RuntimeError, match="supports verl 0.9.1.*LLMServerClient"):
        requests.patch_server_client(types.SimpleNamespace(__name__="verl.workers.rollout.llm_server"))


def test_a_missing_manager_method_fails_clearly():
    module = types.SimpleNamespace(LLMServerClient=type("LLMServerClient", (), {"generate": None}))
    with pytest.raises(RuntimeError, match="supports verl 0.9.1.*_vllm_request_id"):
        requests.patch_server_client(module)


def test_the_span_records_the_weights_and_the_trajectory_of_the_request(spans):
    module = _module()

    async def generate(self, request_id, **kwargs):
        self._vllm_request_id(request_id)
        # VERL's client tags the output with the weights it was generated with.
        return types.SimpleNamespace(extra_fields={"global_steps": 4, "min_global_steps": 3, "max_global_steps": 4})

    module.LLMServerClient.generate = generate
    requests.patch_server_client(module)
    client = module.LLMServerClient()

    class AgentLoopWorker:
        async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name="single_turn", **kwargs):
            return await client.generate("traj", prompt_ids=[], sampling_params=sampling_params)

    agent_loop = types.SimpleNamespace(AgentLoopWorker=AgentLoopWorker)
    assert requests.patch_agent_loop(agent_loop) is True
    assert requests.patch_agent_loop(agent_loop) is False

    import numpy as np

    trajectory = {"step": np.int64(5), "sample_index": np.int64(17), "rollout_n": 2, "validate": False}
    asyncio.run(AgentLoopWorker()._run_agent_loop({}, trajectory=trajectory))

    [(_, kwargs)] = spans
    attributes = kwargs["attributes"]
    assert (attributes["weight_version"], attributes["min_weight_version"], attributes["max_weight_version"]) == (
        4,
        3,
        4,
    )
    assert (attributes["batch_step"], attributes["sample_index"], attributes["rollout_n"]) == (5, 17, 2)
    assert attributes["validate"] is False
    assert type(attributes["batch_step"]) is int, "numpy scalars must be plain for the JSON trace"


def test_one_weight_version_is_not_repeated(spans):
    module = _module()

    async def generate(self, request_id, **kwargs):
        self._vllm_request_id(request_id)
        return types.SimpleNamespace(extra_fields={"global_steps": 4, "min_global_steps": 4, "max_global_steps": 4})

    module.LLMServerClient.generate = generate
    requests.patch_server_client(module)
    asyncio.run(module.LLMServerClient().generate("traj", prompt_ids=[], sampling_params={}))

    [(_, kwargs)] = spans
    assert kwargs["attributes"]["weight_version"] == 4
    assert "min_weight_version" not in kwargs["attributes"] and "batch_step" not in kwargs["attributes"]
