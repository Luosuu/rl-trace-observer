"""Merge real RL-Insight, VERL Torch Profiler and VizTracer output and load it in Perfetto."""

import rl_insight
import torch
from verl.utils.profiler.torch_profile import get_torch_profiler

from rl_trace_observer.integrations.rl_insight import register_rl_insight_client
from rl_trace_observer.integrations.verl.viztracer import VizTracerObserver
from rl_trace_observer.merger.cli import main
from rl_trace_observer.observer import ObserverContext

# Each profiler anchors on its own wall-clock reading; within one process they
# must agree far better than this.
TOLERANCE_NS = 5_000_000


def _matmul_workload():
    x = torch.randn(128, 128)
    for _ in range(3):
        x = x @ x
    return x


def _record_one_step(output_dir):
    context = ObserverContext(
        rank=0,
        tool=None,
        config=None,
        tool_config=None,
        save_file_prefix="actor",
        metadata={"role": "e2e", "profile_step": 3},
    )
    viztracer = VizTracerObserver()
    torch_profiler = get_torch_profiler(
        contents=[], save_path=str(output_dir), role="train", save_file_prefix="actor", rank=0, profile_step=3
    )
    register_rl_insight_client()
    rl_insight.init()
    try:
        viztracer.on_start(context)
        torch_profiler.start()
        with rl_insight.trace_state("actor_update", state_lane_id="rank_0"):
            with torch.profiler.record_function("actor_update"):
                _matmul_workload()
        torch_profiler.stop()
        viztracer.on_stop(context)
    finally:
        rl_insight.finish()


def test_real_sources_are_aligned_in_perfetto(tmp_path, monkeypatch, load_in_perfetto):
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("RL_TRACE_OUTPUT_DIR", str(artifacts))
    monkeypatch.setenv("RL_INSIGHT_SERVER_URL", "local://rl-trace-observer")
    monkeypatch.setenv("RL_TRACE_VIZTRACER_MIN_DURATION_US", "0")
    _record_one_step(artifacts)

    merged = tmp_path / "merged.json"
    assert main([str(artifacts), "-o", str(merged), "--strict"]) == 0

    processor = load_in_perfetto(merged)
    assert (
        processor.query("select * from stats where severity in ('error', 'data_loss') and value > 0")
        .as_pandas_dataframe()
        .empty
    )

    slices = processor.query(
        """
        select s.name, s.ts, s.dur, t.name as thread, t.upid
        from slice s
        join thread_track tt on s.track_id = tt.id
        join thread t using (utid)
        where s.name = 'actor_update' or s.name like '_matmul_workload%'
        """
    ).as_pandas_dataframe()
    by_source = {
        prefix: slices[slices.thread.str.startswith(prefix)].iloc[0] for prefix in ("RL-Insight", "Torch", "VizTracer")
    }
    # The process record ties all three profilers to the one OS process that ran them.
    assert len({span.upid for span in by_source.values()}) == 1

    rl_insight_span = by_source["RL-Insight"]
    for source in ("Torch", "VizTracer"):
        span = by_source[source]
        assert span.dur > 0
        assert rl_insight_span.ts - TOLERANCE_NS <= span.ts
        assert span.ts + span.dur <= rl_insight_span.ts + rl_insight_span.dur + TOLERANCE_NS
    assert abs(by_source["Torch"].ts - rl_insight_span.ts) < TOLERANCE_NS
