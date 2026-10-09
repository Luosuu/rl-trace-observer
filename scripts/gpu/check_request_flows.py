"""P6-c: follow rollout requests through a merged step trace, down to GPU kernels.

    python scripts/gpu/check_request_flows.py STEP_TRACE [--out summary.json]

For every ``tokenspeed_generate`` (one per request a server handled in the
step), counts how far its request flow reaches:

1. from the agent loop's ``rollout_request`` with the same ``request_id``;
2. into ``forward_batch`` slices that list the request (first and last forward);
3. from those forwards to Proton GPU kernels: a slice inside the forward starts
   a VizTracer->Proton scope flow (an eager kernel's scope, or the CUDA graph
   replay's) and Proton links that scope, or a scope inside it, to kernels.

Also checks that no request flow passes through a forward that does not list
its request. Prints and writes a summary; exits non-zero if a check fails.
"""

import argparse
import json
import sys
from collections import defaultdict
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

    points = query(
        """
        select s.id, s.name, extract_arg(s.arg_set_id, 'args.request_id') as request_id
        from slice s where s.name in ('rollout_request', 'tokenspeed_generate')
        """
    )
    forwards = query(
        """
        select s.id, a.string_value as request_id
        from slice s join args a on a.arg_set_id = s.arg_set_id
        where s.name = 'forward_batch' and a.key glob 'args.request_ids*'
        """
    )
    served = defaultdict(set)
    for slice_id, request_id in zip(forwards.id, forwards.request_id, strict=True):
        served[slice_id].add(request_id)
    names = dict(zip(points.id, points.name, strict=True))
    request_of = dict(zip(points.id, points.request_id, strict=True))
    flows = query("select slice_out, slice_in from flow")
    edges = defaultdict(list)
    for out, into in zip(flows.slice_out, flows.slice_in, strict=True):
        edges[out].append(into)

    generates = {slice_id for slice_id, name in names.items() if name == "tokenspeed_generate"}
    callers = {
        request_of[s]
        for s, name in names.items()
        if name == "rollout_request"
        for t in edges[s]
        if t in generates and request_of[t] == request_of[s]
    }

    # Walk each request's flow from its server span through the forwards it reaches.
    reached: dict[str, list[int]] = {}
    crossed = 0
    for generate in generates:
        request_id, chain, current = request_of[generate], [], generate
        while True:
            following = [t for t in edges[current] if t in served]
            if not following:
                break
            current = following[0]
            if request_id not in served[current]:
                crossed += 1
            chain.append(current)
        reached[request_id] = chain

    kernel_cache: dict[int, bool] = {}

    def reaches_kernels(forward: int) -> bool:
        if forward not in kernel_cache:
            found = query(
                f"""
                with scopes as (
                  select f.slice_in as id from descendant_slice({forward}) d join flow f on f.slice_out = d.id
                ), proton as (
                  select id from scopes union select d.id from scopes join descendant_slice(scopes.id) d
                )
                select count(*) as n from proton join flow f on f.slice_out = proton.id
                join slice k on k.id = f.slice_in join thread_track tt on k.track_id = tt.id
                join thread t on t.utid = tt.utid where t.name like 'Proton · GPU%'
                """
            )
            kernel_cache[forward] = bool(found.n[0])
        return kernel_cache[forward]

    with_forwards = [chain for chain in reached.values() if chain]
    summary = {
        "requests": len(generates),
        "linked_from_agent_loop": len(callers),
        "reaching_forwards": len(with_forwards),
        "reaching_two_forwards": sum(len(chain) >= 2 for chain in with_forwards),
        "first_forward_reaches_kernels": sum(reaches_kernels(chain[0]) for chain in with_forwards),
        "last_forward_reaches_kernels": sum(reaches_kernels(chain[-1]) for chain in with_forwards),
        "forwards_not_serving_their_request": crossed,
    }
    checks = {
        "every_request_linked_from_the_agent_loop": summary["linked_from_agent_loop"] == summary["requests"] > 0,
        "every_request_reaches_its_forwards": summary["reaching_forwards"] == summary["requests"],
        "no_flow_crosses_requests": crossed == 0,
        "some_request_reaches_kernels": summary["first_forward_reaches_kernels"] > 0,
    }
    summary["checks"] = checks
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
    processor.close()
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
