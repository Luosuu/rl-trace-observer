"""Write test artifacts the way traced processes do: each file registered in its process record."""

import gzip
import json
from pathlib import Path

from rl_trace_observer.context import STEP_MARKER_ATTRIBUTE
from rl_trace_observer.process_record import SCHEMA_VERSION

BASE_NS = 1_790_000_000_000_000_000
BASE_US = BASE_NS // 1000
MARKER = {STEP_MARKER_ATTRIBUTE: True}


def span(name, start_us=BASE_US, dur_us=5.0, *, pid=10, lane="rank_0", **args):
    return json.dumps(
        {"name": name, "ph": "X", "ts": start_us, "dur": dur_us, "pid": str(pid), "tid": lane, "args": args}
    )


def step_marker(step, start_us=BASE_US, dur_us=1000, *, pid=10, **args):
    """The span the trainer records around one training step."""
    return span("global_step", start_us, dur_us, pid=pid, lane="trainer", global_step=step, **args, **MARKER)


class TracedProcess:
    """One traced OS process: writes artifacts and registers them in its record."""

    def __init__(self, root: Path, pid=10, *, host="node-1", run_id="run-a", role=None, rank=None, actor=None):
        self.root, self.pid, self.host = root, pid, host
        self.record = {
            "schema_version": SCHEMA_VERSION,
            "hostname": f"{host}.example",
            "host": host,
            "os_pid": pid,
            "run_id": run_id,
            "role": role,
            "ray": {"actor_name": actor} if actor else None,
            "torch_distributed": {"rank": rank, "world_size": 2} if rank is not None else None,
            "artifacts": [],
        }
        self.record_path = root / f"rl-trace-process-{host}-pid-{pid}.json"
        self.write_record()

    @property
    def key(self) -> str:
        run_id = self.record["run_id"]
        return f"{self.host}:{self.pid}" if run_id is None else f"{self.host}:{self.pid}@{run_id}"

    def write_record(self) -> None:
        self.record_path.write_text(json.dumps(self.record))

    def register(self, path: Path, kind: str, **fields) -> Path:
        self.record["artifacts"].append({"kind": kind, "file": path.name, **fields})
        self.write_record()
        return path

    def jsonl(self, lines=None, *, register=True) -> Path:
        path = self.root / f"rl-insight-{self.host}-pid-{self.pid}.chrome.jsonl"
        lines = [span("actor_update", pid=self.pid)] if lines is None else lines
        path.write_text("".join(line + "\n" for line in lines))
        return self.register(path, "rl_insight_jsonl") if register else path

    def torch(self, step=1, *, base_ns=BASE_NS, register=True) -> Path:
        path = self.root / f"actor_train_step{step}_rank0-of-2_pid{self.pid}_2026093015115{step:04d}.json.gz"
        write_torch(path, self.pid, base_ns)
        return self.register(path, "torch", global_step=step, role="actor_train") if register else path

    def tokenspeed(self, rank_tag="TP0", step=1, *, base_ns=BASE_NS, register=True, **kwargs) -> tuple[Path, Path]:
        """The VizTracer report and Proton trace one TokenSpeed scheduler rank wrote for this server actor."""
        profile_id = f"{self.record['run_id']}-step-{step}"
        directory = self.root / "rollout" / "replica0"
        directory.mkdir(parents=True, exist_ok=True)
        files = write_tokenspeed_pair(directory, profile_id, rank_tag, base_ns=base_ns, **kwargs)
        if register:
            for path, kind in zip(files, ("tokenspeed_viztracer", "tokenspeed_proton"), strict=True):
                self.record["artifacts"].append(
                    {
                        "kind": kind,
                        "file": str(path.relative_to(self.root)),
                        "global_step": step,
                        "role": "rollout_replica0",
                        "rank_tag": rank_tag,
                    }
                )
            self.write_record()
        return files


def write_torch(path: Path, pid: int, base_ns=BASE_NS) -> Path:
    data = {
        "baseTimeNanoseconds": base_ns,
        "traceEvents": [
            {"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": "ray::WorkerDict"}},
            {"ph": "X", "name": "actor_update", "pid": pid, "tid": pid, "ts": 1.0, "dur": 3.0},
            {"ph": "X", "name": "gemm", "pid": 0, "tid": 7, "ts": 2.0, "dur": 1.0},
        ],
    }
    with gzip.open(path, "wt") as file:
        json.dump(data, file)
    return path


def two_step_run(root: Path, run_id="run-a") -> tuple[TracedProcess, TracedProcess]:
    """A trainer marking steps 1 and 2 and an actor with spans and a Torch trace in each."""
    trainer = TracedProcess(root, 10, run_id=run_id, role="trainer")
    trainer.jsonl([step_marker(1, pid=10, run_id=run_id), step_marker(2, BASE_US + 2000, pid=10, run_id=run_id)])
    actor = TracedProcess(root, 20, run_id=run_id, rank=0)
    actor.jsonl([span("actor_update", BASE_US + 100, 10, pid=20), span("actor_update", BASE_US + 2100, 10, pid=20)])
    actor.torch(1)
    actor.torch(2, base_ns=BASE_NS + 2_000_000)
    return trainer, actor


def write_tokenspeed_pair(
    directory: Path, profile_id: str, rank_tag="TP0", *, base_ns=BASE_NS, proton_offset_ns=2_000, scheduler_pid=900
) -> tuple[Path, Path]:
    """Files shaped like a TokenSpeed scheduler's ``/start_profile`` VIZTRACER + PROTON output.

    The VizTracer report runs ``forward`` at 10-30 µs and starts a scope flow
    for scope 5 (recorded by Proton) and scope 6 (not recorded). Proton, anchored
    ``proton_offset_ns`` later, records scope 5 on its CPU thread and the
    kernel it launched on a GPU stream, linked by its own flow 1.
    """
    viztracer = directory / f"{profile_id}-{rank_tag}.viztracer.json"
    viztracer.write_text(
        json.dumps(
            {
                "viztracer_metadata": {"version": "1.1.1", "baseTimeNanoseconds": base_ns},
                "traceEvents": [
                    {"ph": "M", "name": "process_name", "pid": scheduler_pid, "tid": 0, "args": {"name": "scheduler"}},
                    {"ph": "M", "name": "thread_name", "pid": scheduler_pid, "tid": 1, "args": {"name": "MainThread"}},
                    {"ph": "X", "name": "forward", "pid": scheduler_pid, "tid": 1, "ts": 10.0, "dur": 20.0},
                    *(
                        {"ph": "s", "name": "viztracer->proton", "cat": "tokenspeed.proton", "id": scope}
                        | {"pid": scheduler_pid, "tid": 1, "ts": 11.0}
                        for scope in (5, 6)
                    ),
                ],
            }
        )
    )
    offset_us = proton_offset_ns / 1000
    proton = directory / f"{profile_id}-{rank_tag}.proton.chrome_trace"
    proton.write_text(
        json.dumps(
            {
                "baseTimeNanoseconds": base_ns + proton_offset_ns,
                "displayTimeUnit": "ns",
                "traceEvents": [
                    {"ph": "M", "name": "process_name", "pid": 0, "tid": 0, "args": {"name": "Trace"}},
                    {"ph": "M", "name": "thread_name", "pid": 0, "tid": 1, "args": {"name": "CPU Thread 900"}},
                    {"ph": "M", "name": "thread_name", "pid": 0, "tid": 2, "args": {"name": "GPU Stream 7"}},
                    {"ph": "X", "name": "attention", "pid": 0, "tid": 1, "ts": 12.0 - offset_us, "dur": 4.0}
                    | {"args": {"scope_id": 5}},
                    {"ph": "X", "name": "flash_attn_kernel", "pid": 0, "tid": 2, "ts": 20.0 - offset_us, "dur": 6.0},
                    {"ph": "s", "name": "launch", "pid": 0, "tid": 1, "ts": 13.0 - offset_us, "id": 1},
                    {"ph": "f", "name": "launch", "pid": 0, "tid": 2, "ts": 20.0 - offset_us, "id": 1, "bp": "e"},
                ],
            }
        )
    )
    return viztracer, proton
