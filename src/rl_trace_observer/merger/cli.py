"""``rl-trace-merge``: merge per-process artifacts into one Perfetto trace."""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from .manifest import build_manifest
from .merge import EmptyTraceError, merge_sources

logger = logging.getLogger("rl_trace_observer.merger")


def _default_manifest_path(output: Path) -> Path:
    stem = output.name.removesuffix(".gz").removesuffix(".json")
    return output.with_name(f"{stem}.manifest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rl-trace-merge",
        description="Merge RL-Insight JSONL, Torch Profiler and VizTracer artifacts into one Chrome Trace.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="artifact files or directories (searched recursively)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="merged Chrome Trace JSON to write")
    parser.add_argument(
        "--manifest", type=Path, help="session manifest to write (default: <output stem>.manifest.json)"
    )
    parser.add_argument(
        "--strict", action="store_true", help="fail on any manifest problem or dropped event instead of warning"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest, sources = build_manifest(args.inputs)
    for problem in manifest.problems:
        logger.warning("%s: %s (%s)", problem.kind, problem.path, problem.detail)

    result = None
    if not sources:
        logger.error("No readable trace artifacts found in %s", ", ".join(map(str, args.inputs)))
    else:
        try:
            result = merge_sources(sources, manifest.processes)
        except EmptyTraceError as error:
            logger.error("%s in %s", error, ", ".join(map(str, args.inputs)))
    if result is not None:
        for warning in result.warnings:
            logger.warning(warning)
        if args.strict and (manifest.problems or result.warnings):
            logger.error(
                "Strict mode: %d manifest problems, %d merge warnings", len(manifest.problems), len(result.warnings)
            )
            result = None

    # The manifest is written even when merging fails: its problems explain why.
    # A trace left from an earlier run must not sit next to it.
    if result is None and args.output.exists():
        logger.warning("Removing %s from an earlier run", args.output)
        args.output.unlink()
    manifest_path = args.manifest or _default_manifest_path(args.output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            manifest.to_json(
                global_base_time_ns=result.global_base_ns if result else None,
                output=str(args.output) if result else None,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if result is None:
        logger.error("No trace written; manifest: %s", manifest_path)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result.trace, file, separators=(",", ":"))
    counts = Counter(source.kind for source in sources)
    logger.info(
        "Merged %s from %d processes into %s (%d events); manifest: %s",
        ", ".join(f"{count} {kind}" for kind, count in sorted(counts.items())),
        len(manifest.processes),
        args.output,
        len(result.trace["traceEvents"]),
        manifest_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
