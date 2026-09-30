"""``rl-trace-merge``: merge per-process artifacts into one Perfetto trace."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .merge import merge_sources
from .sources import discover, read_source

logger = logging.getLogger("rl_trace_observer.merger")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rl-trace-merge",
        description="Merge RL-Insight JSONL, Torch Profiler and VizTracer artifacts into one Chrome Trace.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="artifact files or directories (searched recursively)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="merged Chrome Trace JSON to write")
    parser.add_argument("--strict", action="store_true", help="fail instead of warning on dropped events")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    artifacts = discover(args.inputs)
    if not artifacts:
        logger.error("No trace artifacts found in %s", ", ".join(map(str, args.inputs)))
        return 1

    sources = [read_source(kind, path) for kind, path in artifacts]
    result = merge_sources(sources)
    for warning in result.warnings:
        logger.warning(warning)
    if args.strict and result.warnings:
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result.trace, file, separators=(",", ":"))
    counts = {kind: sum(1 for k, _ in artifacts if k == kind) for kind in sorted({k for k, _ in artifacts})}
    logger.info(
        "Merged %s into %s (%d events)",
        ", ".join(f"{count} {kind}" for kind, count in counts.items()),
        args.output,
        len(result.trace["traceEvents"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
