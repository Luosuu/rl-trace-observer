"""A stand-in for ``tokenspeed serve`` with TP=1, for CPU tests of the TokenSpeed rollout.

Launched through ``RL_TRACE_TOKENSPEED_COMMAND`` with ``tokenspeed serve``'s
arguments, it serves on ``--control-port`` the routes the integration uses:

* ``/generate`` (SGLang style): random tokens with log probs at a fixed pace;
* ``/release_memory_occupation`` / ``/resume_memory_occupation``, enforced:
  generating without KV cache or loading weights while they are released fails;
* ``/init_weights_update_group`` / ``/update_weights_from_distributed``: joins
  the trainer's group at ``rank_offset`` and receives every broadcast tensor;
* ``/start_profile`` / ``/stop_profile``: writes a VizTracer report and a
  Proton Chrome trace named and anchored like TokenSpeed's, rank 0 after a
  delay, as TokenSpeed saves VizTracer reports after replying.

Every request is appended to ``$FAKE_TOKENSPEED_LOG_DIR/fake-tokenspeed-<port>.jsonl``.
"""

import argparse
import json
import math
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

SECONDS_PER_TOKEN = 0.01


class State:
    def __init__(self, args):
        config = json.loads((Path(args.model) / "config.json").read_text())
        self.vocab_size = int(config["vocab_size"])
        eos = config.get("eos_token_id")
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos
        self.released: set[str] = set()
        self.group = None
        self.profile: tuple[str, str, int] | None = None
        self.lock = threading.Lock()
        log_dir = Path(os.environ.get("FAKE_TOKENSPEED_LOG_DIR", "."))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"fake-tokenspeed-{args.control_port}.jsonl"

    def log(self, **event):
        with self.lock, self.log_path.open("a") as file:
            file.write(json.dumps({"time_ns": time.time_ns(), **event}) + "\n")


def _join_group(body):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert 1 <= body["rank_offset"] < body["world_size"], body
    store, rank, world_size = next(
        rendezvous(
            f"tcp://{body['master_address']}:{body['master_port']}",
            body["rank_offset"],
            body["world_size"],
            timeout=default_pg_timeout,
        )
    )
    group, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend(body["backend"]),
        PrefixStore(body["group_name"], store),
        group_name=body["group_name"],
        backend_options=None,
        timeout=default_pg_timeout,
    )
    _world.pg_group_ranks[group] = {i: i for i in range(world_size)}
    return group


def _write_profile(output_dir: Path, profile_id: str, start_ns: int, rank: int) -> None:
    """One rank's VizTracer report and Proton trace over [start_ns, now]."""
    elapsed_us = (time.time_ns() - start_ns) / 1000
    tag = f"TP{rank}"
    pid = os.getpid()
    viztracer = {
        "viztracer_metadata": {"version": "fake", "baseTimeNanoseconds": start_ns},
        "traceEvents": [
            {"ph": "M", "name": "thread_name", "pid": pid, "tid": 1, "args": {"name": "MainThread"}},
            {"ph": "X", "name": "event_loop", "pid": pid, "tid": 1, "ts": 0.0, "dur": elapsed_us},
            {"ph": "X", "name": "forward", "pid": pid, "tid": 1, "ts": 10.0, "dur": 20.0},
            {"ph": "s", "name": "viztracer->proton", "cat": "tokenspeed.proton", "id": 1}
            | {"pid": pid, "tid": 1, "ts": 11.0},
        ],
    }
    # Proton's clock starts 1 µs later.
    proton = {
        "baseTimeNanoseconds": start_ns + 1000,
        "traceEvents": [
            {"ph": "M", "name": "process_name", "pid": 0, "tid": 0, "args": {"name": "Trace"}},
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 1, "args": {"name": f"CPU Thread {pid}"}},
            {"ph": "M", "name": "thread_name", "pid": 0, "tid": 2, "args": {"name": "GPU Stream 7"}},
            {"ph": "X", "name": "attention", "pid": 0, "tid": 1, "ts": 11.0, "dur": 4.0, "args": {"scope_id": 1}},
            {"ph": "X", "name": "attention_kernel", "pid": 0, "tid": 2, "ts": 19.0, "dur": 6.0},
            {"ph": "s", "name": "launch", "pid": 0, "tid": 1, "ts": 12.0, "id": 1},
            {"ph": "f", "name": "launch", "pid": 0, "tid": 2, "ts": 19.0, "id": 1, "bp": "e"},
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{profile_id}-{tag}.proton.chrome_trace").write_text(json.dumps(proton))
    temporary = output_dir / f".{profile_id}-{tag}.viztracer.json.tmp"
    temporary.write_text(json.dumps(viztracer))
    os.replace(temporary, output_dir / f"{profile_id}-{tag}.viztracer.json")


def make_handler(state: State, tp: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, payload, status=200):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _fail(self, message):
            state.log(event="error", path=self.path, message=message)
            self._reply({"success": False, "message": message}, status=400)

        def do_GET(self):
            if self.path == "/health":
                self._reply({"status": "ok"})
            elif self.path == "/flush_cache":
                state.log(event="flush_cache")
                self._reply({"success": True})
            else:
                self._reply({"error": self.path}, status=404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            route = getattr(self, "route_" + self.path.strip("/"), None)
            if route is None:
                self._reply({"error": self.path}, status=404)
            else:
                route(body)

        def route_generate(self, body):
            if state.released:
                return self._fail(f"generate while {sorted(state.released)} released")
            params = body["sampling_params"]
            max_tokens = int(params.get("max_new_tokens") or 8)
            rng = random.Random(f"{body['rid']}:{len(body['input_ids'])}")
            length = rng.randint(1, max(1, max_tokens))
            token_ids = [rng.randrange(8, state.vocab_size) for _ in range(length)]
            finish = "length"
            if state.eos_token_id is not None and length < max_tokens:
                token_ids.append(state.eos_token_id)
                finish = "stop"
            time.sleep(SECONDS_PER_TOKEN * len(token_ids))
            logprob = -math.log(state.vocab_size)
            meta_info = {"finish_reason": {"type": finish}}
            if body.get("return_logprob"):
                meta_info["output_token_logprobs"] = [[logprob, token, None] for token in token_ids]
            state.log(event="generate", rid=body["rid"], tokens=len(token_ids))
            self._reply({"output_ids": token_ids, "meta_info": meta_info})

        def route_release_memory_occupation(self, body):
            state.released |= set(body.get("tags") or ["kv_cache", "weights"])
            state.log(event="release", tags=body.get("tags"))
            self._reply({"success": True})

        def route_resume_memory_occupation(self, body):
            state.released -= set(body.get("tags") or ["kv_cache", "weights"])
            state.log(event="resume", tags=body.get("tags"))
            self._reply({"success": True})

        def route_abort_request(self, body):
            state.log(event="abort", body=body)
            self._reply({"success": True})

        def route_init_weights_update_group(self, body):
            state.group = _join_group(body)
            state.log(
                event="init_group",
                backend=body["backend"],
                group_name=body["group_name"],
                rank_offset=body["rank_offset"],
                world_size=body["world_size"],
            )
            self._reply({"success": True, "message": "joined"})

        def route_update_weights_from_distributed(self, body):
            if state.group is None:
                return self._fail("weight update group not initialized")
            if "weights" in state.released:
                return self._fail("weights are released")
            total = 0.0
            for dtype, shape in zip(body["dtypes"], body["shapes"], strict=True):
                buffer = torch.empty(shape, dtype=getattr(torch, dtype))
                torch.distributed.broadcast(buffer, src=0, group=state.group)
                total += float(buffer.float().sum())
            state.log(event="update", names=body["names"], checksum=total)
            self._reply({"success": True, "message": f"updated {len(body['names'])} weights"})

        def route_start_profile(self, body):
            if state.profile is not None:
                return self._fail("Profiling is already in progress")
            state.profile = (body["output_dir"], body["profile_id"], time.time_ns())
            state.log(event="start_profile", body=body)
            self._reply({"success": True, "message": "Succeeded"})

        def route_stop_profile(self, body):
            if state.profile is None:
                return self._fail("Profiling is not in progress")
            output_dir, profile_id, start_ns = state.profile
            state.profile = None
            for rank in range(1, tp):
                _write_profile(Path(output_dir), profile_id, start_ns, rank)
            # Rank 0 finishes after the reply.
            threading.Timer(1.0, _write_profile, (Path(output_dir), profile_id, start_ns, 0)).start()
            state.log(event="stop_profile", profile_id=profile_id)
            self._reply({"success": True, "message": "Succeeded."})

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--control-port", type=int, required=True)
    args, _ = parser.parse_known_args()
    state = State(args)
    state.log(event="launch", args=vars(args))
    ThreadingHTTPServer(("0.0.0.0", args.control_port), make_handler(state, args.tp)).serve_forever()


if __name__ == "__main__":
    main()
