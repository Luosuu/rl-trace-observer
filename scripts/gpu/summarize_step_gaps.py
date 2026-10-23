"""Where a profiled step's time goes after its GPU work: profiler stops and rollout pickup.

    python scripts/gpu/summarize_step_gaps.py STEP_TRACE [--out gaps.json]

Times are ms from the start of the step's ``global_step`` span:

- ``trainer``: when the training workers' last kernel ended, each worker's
  ``torch_profiler_stop`` (and its parts, with garbage collections meanwhile),
  its ``torch_trace_export`` in the background, and the driver's ``stop_profile``;
- ``rollout``: when the servers' last kernel ended, when each agent loop
  worker finished its share of the batch (``agent_loop_batch``) and when the
  trainer picked the batch up (the end of ``rollout_batch``, one-step-off only);
- ``tail_ms``: from the last kernel anywhere to the end of the step.

Prints and writes the summary.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    from perfetto.trace_processor import TraceProcessor

    processor = TraceProcessor(trace=str(args.trace))

    def query(sql: str):
        return processor.query(sql).as_pandas_dataframe()

    [window] = query("select ts, dur from slice where name = 'global_step'").itertuples()
    start = int(window.ts)

    def ms(value) -> float:
        return round((float(value) - start) / 1e6, 1)

    def last_kernel(thread_glob: str):
        found = query(
            f"""
            select max(s.ts + s.dur) as last from slice s join thread_track tt on s.track_id = tt.id
            join thread t on t.utid = tt.utid where t.name glob '{thread_glob}'
            """
        ).last[0]
        return None if found != found or found is None else ms(found)

    def spans(name: str, *args: str):
        extracted = "".join(f", extract_arg(arg_set_id, 'args.{arg}') as {arg}" for arg in args)
        return query(f"select ts, dur{extracted} from slice where name = '{name}' order by ts")

    stops = spans(
        "torch_profiler_stop",
        "backend",
        "total_ms",
        "disable_ms",
        "export_dispatch_ms",
        "export_copy_ms",
        "export_thread_start_ms",
        "gc_ms",
        "other_ms",
    )
    driver_stops = spans("stop_profile")
    exports = spans("torch_trace_export")
    batches = spans("agent_loop_batch", "samples", "batch_step")
    pickups = spans("rollout_batch", "batch_step", "consumed_by_step")
    trainer_last = last_kernel("Torch · stream *")
    rollout_last = last_kernel("Proton · GPU Stream *")
    summary = {
        "step_ms": round(float(window.dur) / 1e6, 1),
        "trainer": {
            "last_kernel_end_ms": trainer_last,
            "torch_profiler_stops": [
                {
                    "start_ms": ms(s.ts),
                    "backend": s.backend,
                    **{
                        key: getattr(s, key)
                        for key in (
                            "total_ms",
                            "disable_ms",
                            "export_dispatch_ms",
                            "export_copy_ms",
                            "export_thread_start_ms",
                            "gc_ms",
                            "other_ms",
                        )
                        if getattr(s, key) == getattr(s, key)
                    },
                }
                for s in stops.itertuples()
            ],
            "torch_trace_exports": [
                {"start_ms": ms(e.ts), "end_ms": ms(e.ts + e.dur), "dur_ms": round(e.dur / 1e6, 1)}
                for e in exports.itertuples()
            ],
            "driver_stop_profile": [
                {"start_ms": ms(s.ts), "end_ms": ms(s.ts + s.dur), "dur_ms": round(s.dur / 1e6, 1)}
                for s in driver_stops.itertuples()
            ],
        },
        "rollout": {
            "last_kernel_end_ms": rollout_last,
            "agent_loop_batches_end_ms": sorted(ms(b.ts + b.dur) for b in batches.itertuples()),
            "rollout_batches": [{"batch_step": b.batch_step, "end_ms": ms(b.ts + b.dur)} for b in pickups.itertuples()],
        },
    }
    ends = [end for end in (trainer_last, rollout_last) if end is not None]
    if ends:
        summary["tail_ms"] = round(summary["step_ms"] - max(ends), 1)
    if len(batches) and len(pickups):
        generated = max(ms(b.ts + b.dur) for b in batches.itertuples())
        summary["rollout"]["pickup_delay_ms"] = round(
            max(ms(p.ts + p.dur) for p in pickups.itertuples()) - generated, 1
        )
    print(json.dumps(summary, indent=2, default=str))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2, default=str))
    processor.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
