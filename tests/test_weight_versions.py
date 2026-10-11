"""The rollout servers' weight version: one weight_update span per update, a counter track when merged."""

import asyncio
import json
import sys
import types

import pytest

from rl_trace_observer.context import WEIGHT_UPDATE_ATTRIBUTE
from rl_trace_observer.integrations.verl import weights
from rl_trace_observer.merger import merge_sources
from rl_trace_observer.merger.manifest import build_manifest, select


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
    from verl.utils.ray_utils import auto_await

    class CheckpointEngineManager:
        """The shape of VERL's manager: ``update_weights`` runs or returns its coroutine, as its caller needs."""

        def __init__(self):
            self.updates = []

        @auto_await
        async def update_weights(self, global_steps: int = None):
            await asyncio.sleep(0.01)
            self.updates.append(global_steps)
            return {"synced": global_steps}

    return types.SimpleNamespace(CheckpointEngineManager=CheckpointEngineManager)


def test_each_update_records_the_version_it_installs(spans):
    module = _module()
    assert weights.patch_checkpoint_engine_manager(module) is True
    assert weights.patch_checkpoint_engine_manager(module) is False
    manager = module.CheckpointEngineManager()

    # From a sync trainer (no event loop), and awaited from an async one.
    assert manager.update_weights(0) == {"synced": 0}

    async def fit_step():
        return await manager.update_weights(global_steps=1)

    assert asyncio.run(fit_step()) == {"synced": 1}
    # Restoring the same weights is not a new version.
    manager.update_weights()

    assert manager.updates == [0, 1, None]
    assert [(name, kwargs["attributes"]["weight_version"]) for name, kwargs in spans] == [
        ("weight_update", 0),
        ("weight_update", 1),
    ]
    for _, kwargs in spans:
        assert kwargs["attributes"]["state_lane_id"] == "trainer/weights"
        assert kwargs["attributes"][WEIGHT_UPDATE_ATTRIBUTE] is True
        assert kwargs["end_time_ns"] - kwargs["start_time_ns"] >= 10_000_000


def test_a_renamed_manager_fails_clearly():
    with pytest.raises(RuntimeError, match="supports verl 0.9.1.*CheckpointEngineManager"):
        weights.patch_checkpoint_engine_manager(types.SimpleNamespace(__name__="verl.checkpoint_engine.base"))


def _three_steps(root):
    """A sync trainer: step n generates with version n - 1 and updates the servers to n at its end."""
    from trace_artifacts import BASE_US, TracedProcess, span, step_marker

    def update(version, start_us):
        return span(
            "weight_update", start_us, 100, pid=10, lane="trainer/weights", weight_version=version,
            **{WEIGHT_UPDATE_ATTRIBUTE: True},
        )  # fmt: skip

    lines = [update(0, BASE_US - 500)]
    for step in (1, 2, 3):
        start = BASE_US + (step - 1) * 2000
        lines += [step_marker(step, start, 1000), update(step, start + 800)]
    TracedProcess(root, 10, role="trainer").jsonl(lines)


def _counter(trace):
    """(ts in µs from the trace start, version) of each weight version counter event, and the update slices."""
    events = trace["traceEvents"]
    counters = [(e["ts"], e["args"]["weight_version"]) for e in events if e["ph"] == "C"]
    updates = [e["args"]["weight_version"] for e in events if e.get("name") == "weight_update"]
    return sorted(counters), updates


def test_a_step_trace_starts_with_the_weights_the_servers_held(tmp_path):
    _three_steps(tmp_path)
    manifest, sources = build_manifest([tmp_path])

    result = merge_sources(select(manifest, sources, steps=[2]), manifest.processes)

    # Step 2 starts on version 1 (updated at the end of step 1, outside the
    # window) and ends on version 2, which its own update installs.
    counters, updates = _counter(result.trace)
    assert counters == [(0.0, 1), (900.0, 2)]
    assert updates == [2], "the carried update only sets the counter"
    assert manifest.problems == []


def test_the_weight_version_is_a_counter_track_in_perfetto(tmp_path, load_in_perfetto):
    _three_steps(tmp_path)
    manifest, sources = build_manifest([tmp_path])
    merged = tmp_path / "merged.json"
    merged.write_text(json.dumps(merge_sources(select(manifest, sources), manifest.processes).trace))

    values = (
        load_in_perfetto(merged)
        .query(
            """
            select t.name, c.value from counter c join process_counter_track t on c.track_id = t.id
            order by c.ts
            """
        )
        .as_pandas_dataframe()
    )
    assert set(values.name) == {"rollout weight_version"}
    assert list(values.value) == [0, 1, 2, 3]
