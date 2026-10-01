"""E0: TokenSpeed on its own, through the HTTP API the VERL rollout uses.

Launches ``tokenspeed serve`` (TP=2 by default), then checks and records:

1. ``/generate`` with ``input_ids`` returns token ids with one log prob each;
2. a ``VIZTRACER`` + ``PROTON`` profile over concurrent requests writes one
   VizTracer report and one Proton Chrome trace per TP rank, each with its
   ``baseTimeNanoseconds`` anchor;
3. releasing and resuming weights and KV cache keeps generation working;
4. weights broadcast from a "trainer" on the next GPU through
   ``/update_weights_from_distributed`` are loaded, for each way of wrapping
   the update (``SYNC_PROTOCOLS``): zeroing the final norm changes greedy
   output and sending the original weights restores it;
5. ``rl-trace-merge`` and ``tokenspeed merge-traces --all-ranks`` both merge
   the profile.

    python scripts/gpu/tokenspeed_smoke.py --model PATH --out DIR [--tp 2]

Writes ``DIR/summary.json``; exits non-zero if a check fails.
"""

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, model: str, tp: int, log: Path, extra_args: tuple[str, ...] = ()):
        self.port, self.control_port = _free_port(), _free_port()
        env = {
            **os.environ,
            # The server takes the first tp GPUs; the "trainer" uses the next one.
            "CUDA_VISIBLE_DEVICES": ",".join(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2").split(",")[:tp]),
            "TOKENSPEED_KERNEL_PROFILE_DATA": "trace",
            "TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT": "chrome_trace",
        }
        command = [
            sys.executable, "-m", "tokenspeed.cli", "serve",
            "--model", model, "--tp", str(tp),
            "--host", "127.0.0.1", "--port", str(self.port), "--control-port", str(self.control_port),
            "--gpu-memory-utilization", "0.5", "--enable-output-logprobs", "--enable-memory-saver", *extra_args,
        ]  # fmt: skip
        print("launching:", " ".join(command), flush=True)
        self.log = log.open("w")
        self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 1800
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(f"tokenspeed serve exited with {self.process.returncode}; see {log}")
            try:
                if requests.get(self.url("/health"), timeout=5).status_code == 200:
                    break
            except requests.RequestException:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("tokenspeed serve did not become ready")
            time.sleep(2)

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.control_port}{path}"

    def post(self, path: str, body: dict | None = None, timeout: float = 600) -> dict:
        response = requests.post(self.url(path), json=body or {}, timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError(f"POST {path}: {response.status_code} {response.text[:2000]}")
        return response.json() if response.text else {}

    def dump_stacks(self, path: Path) -> None:
        """py-spy stacks of every server process, for a hang."""
        import psutil

        with path.open("a") as file:
            for proc in [psutil.Process(self.process.pid), *psutil.Process(self.process.pid).children(recursive=True)]:
                result = subprocess.run(["py-spy", "dump", "--pid", str(proc.pid)], capture_output=True, text=True)
                file.write(f"=== pid {proc.pid} {' '.join(proc.cmdline())[:200]}\n{result.stdout}{result.stderr}\n")

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.log.close()


def _run(function, timeout: float, name: str, on_timeout=None):
    """Run ``function`` in a thread; raise if it does not finish in time."""
    result: dict = {}

    def target():
        try:
            result["value"] = function()
        except BaseException as error:
            result["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    start = time.time()
    thread.start()
    thread.join(timeout)
    print(f"  {name}: {time.time() - start:.1f}s", flush=True)
    if thread.is_alive():
        if on_timeout:
            on_timeout()
        raise TimeoutError(f"{name} did not finish in {timeout}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


# How a weight update is wrapped: the server state around /update_weights_from_distributed.
# "verl" is the order of VERL's hybrid trainer: sleep, wake the weights, update, wake the KV cache.
SYNC_PROTOCOLS = {
    "awake": ([], []),
    "awake_pause": ([("/pause_generation", None)], [("/continue_generation", None)]),
    "kv_released": (
        [("/release_memory_occupation", ["kv_cache"])],
        [("/resume_memory_occupation", ["kv_cache"])],
    ),
    "full_cycle": (
        [
            ("/release_memory_occupation", ["kv_cache", "weights"]),
            ("/resume_memory_occupation", ["kv_cache", "weights"]),
        ],
        [],
    ),
    "verl": (
        [("/release_memory_occupation", ["kv_cache", "weights"]), ("/resume_memory_occupation", ["weights"])],
        [("/resume_memory_occupation", ["kv_cache"])],
    ),
}


class _WeightSync:
    """One server receiving weights from a "trainer" rank on the next GPU, like TokenSpeedServerAdapter."""

    def __init__(self, model_path: str, tp: int, out: Path, prompt: list[int], protocol: str):
        from rl_trace_observer.integrations.tokenspeed.adapter import _join_group

        self.prompt, self.protocol, self.out, self.tp = prompt, protocol, out, tp
        self.group_name = f"smoke_{protocol}_{tp}"
        self.server = Server(model_path, tp, out / f"weight_sync_{protocol}_tp{tp}.log")
        port = _free_port()
        body = {
            "master_address": "127.0.0.1",
            "master_port": port,
            "rank_offset": 1,
            "world_size": 1 + tp,
            "group_name": self.group_name,
            "backend": "nccl",
        }
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            joined = pool.submit(self.server.post, "/init_weights_update_group", body, 120)
            self.group = _run(
                lambda: self._on_device(_join_group, "127.0.0.1", port, 1 + tp, self.group_name, "nccl"),
                120,
                "join",
                self.dump,
            )
            joined.result(timeout=120)

    def _on_device(self, function, *args):
        import torch

        torch.cuda.set_device(torch.device("cuda", self.tp))
        return function(*args)

    def dump(self):
        self.server.dump_stacks(self.out / f"weight_sync_{self.protocol}_tp{self.tp}_stacks.txt")

    def greedy(self) -> list[int]:
        body = {"input_ids": list(self.prompt), "sampling_params": {"max_new_tokens": 32, "temperature": 0.0}}
        return self.server.post("/generate", body, timeout=120)["output_ids"]

    def _send(self, named: dict) -> None:
        import torch

        names = list(named)
        for start in range(0, len(names), 64):
            bucket = names[start : start + 64]
            body = {
                "names": bucket,
                "dtypes": [str(named[n].dtype).removeprefix("torch.") for n in bucket],
                "shapes": [list(named[n].shape) for n in bucket],
                "group_name": self.group_name,
                "flush_cache": False,
            }
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                received = pool.submit(self.server.post, "/update_weights_from_distributed", body, 120)
                for n in bucket:
                    torch.distributed.broadcast(named[n], src=0, group=self.group)
                torch.cuda.synchronize()
                received.result(timeout=120)

    def sync(self, named: dict) -> None:
        before, after = SYNC_PROTOCOLS[self.protocol]
        for path, tags in before:
            self.server.post(path, {"tags": tags} if tags else None)
        _run(lambda: self._on_device(self._send, named), 120, f"send {self.protocol} tp={self.tp}", self.dump)
        for path, tags in after:
            self.server.post(path, {"tags": tags} if tags else None)
        requests.get(self.server.url("/flush_cache"), timeout=60)


def weight_sync(model_path: str, tp: int, out: Path, prompt: list[int], checks: dict) -> dict:
    """For each protocol, zero the final norm and restore it through the weight-sync API."""
    import torch
    from transformers import AutoModelForCausalLM

    report = {}
    cases = [(protocol, tp) for protocol in SYNC_PROTOCOLS] + [("verl", 1)]
    for protocol, case_tp in cases:
        device = torch.device("cuda", case_tp)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)
        weights = {name: tensor.detach().contiguous() for name, tensor in model.state_dict().items()}
        zeroed = {**weights, "model.norm.weight": torch.zeros_like(weights["model.norm.weight"])}
        name = f"weight_sync_{protocol}_tp{case_tp}"
        sync = None
        try:
            sync = _WeightSync(model_path, case_tp, out, prompt, protocol)
            before = sync.greedy()
            sync.sync(zeroed)
            changed = sync.greedy()
            sync.sync(weights)
            after = sync.greedy()
            report[name] = {"changed": changed != before, "restored": after == before}
            checks[name] = changed != before and after == before
        except Exception as error:
            report[name] = {"error": repr(error)}
            checks[name] = False
        finally:
            if sync is not None:
                sync.server.close()
            del model, weights, zeroed
            torch.cuda.empty_cache()
        print(f"  {name}: {report[name]}", flush=True)
    return report


def check_server(model, tp, out, variant, extra_args, prompt_ids, tokenizer) -> dict:
    """Generation, a VIZTRACER+PROTON profile and sleep/wake on one server configuration."""
    result: dict = {"checks": {}}
    checks = result["checks"]
    server = Server(model, tp, out / f"tokenspeed_serve_{variant}.log", extra_args)

    def generate(ids):
        body = {
            "input_ids": list(ids),
            "sampling_params": {"max_new_tokens": 64, "temperature": 1.0},
            "return_logprob": True,
        }
        return server.post("/generate", body)

    try:
        # 1. token-in, token-out generation with log probs
        output = generate(prompt_ids[0])
        result["generate_example"] = output
        token_ids = output.get("output_ids") or []
        logprobs = (output.get("meta_info") or {}).get("output_token_logprobs") or []
        checks["generate_logprobs"] = bool(token_ids) and len(token_ids) == len(logprobs)
        result["generate_text"] = tokenizer.decode(token_ids)

        # 2. profile concurrent requests
        profile_dir = out / f"profile_{variant}"
        start = time.time()
        result["start_profile"] = server.post(
            "/start_profile",
            {"output_dir": str(profile_dir), "activities": ["VIZTRACER", "PROTON"], "profile_id": "e0"},
        )
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            outputs = list(pool.map(generate, prompt_ids))
        try:
            result["stop_profile"] = server.post("/stop_profile")
        except RuntimeError as error:
            result["stop_profile_error"] = str(error)
        result["profile_seconds"] = time.time() - start
        checks["profiled_requests"] = all(o.get("output_ids") for o in outputs)
        expected = {
            f"e0-TP{rank}.{suffix}" for rank in range(tp) for suffix in ("viztracer.json", "proton.chrome_trace")
        }
        deadline = time.monotonic() + 60
        while not expected <= {p.name for p in profile_dir.glob("*")} and time.monotonic() < deadline:
            time.sleep(1)
        files = sorted(p.name for p in profile_dir.glob("*"))
        result["profile_files"] = {name: (profile_dir / name).stat().st_size for name in files}
        checks["profile_files"] = expected <= set(files)
        anchors = {}
        for name in sorted(expected & set(files)):
            data = json.loads((profile_dir / name).read_text())
            anchor = data.get("viztracer_metadata", {}).get("baseTimeNanoseconds") or data.get("baseTimeNanoseconds")
            events = data.get("traceEvents", [])
            anchors[name] = {
                "baseTimeNanoseconds": anchor,
                "events": len(events),
                "flows": sum(1 for e in events if e.get("ph") in ("s", "f")),
                "scope_ids": sum(1 for e in events if isinstance(e.get("args"), dict) and "scope_id" in e["args"]),
            }
        result["profile_anchors"] = anchors
        checks["profile_anchors"] = bool(anchors) and all(a["baseTimeNanoseconds"] for a in anchors.values())

        # 3. sleep and wake
        result["release"] = server.post("/release_memory_occupation", {"tags": ["kv_cache", "weights"]})
        result["resume"] = server.post("/resume_memory_occupation", {"tags": ["kv_cache", "weights"]})
        checks["generate_after_wake"] = bool(generate(prompt_ids[1]).get("output_ids"))
    finally:
        server.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=2)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"checks": {}}
    checks = summary["checks"]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = [f"What is {a} + {b}? Answer briefly." for a, b in zip(range(1, 33), range(100, 132), strict=True)]
    prompt_ids = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)[
            "input_ids"
        ]
        for p in prompts
    ]

    # 1-3 per server configuration: Proton's trace mode may not cover kernels
    # launched by CUDA graph replays, so profiling is also tried without graphs.
    profiled = []
    for variant, extra_args in (("default", ()), ("eager", ("--enforce-eager",))):
        try:
            result = check_server(args.model, args.tp, out, variant, extra_args, prompt_ids, tokenizer)
        except Exception as error:
            result = {"error": repr(error), "checks": {}}
        summary[variant] = result
        checks.update({f"{variant}_{name}": ok for name, ok in result["checks"].items()})
        if result["checks"].get("profile_files"):
            profiled.append(variant)

    # 4. weight sync
    try:
        summary["weight_sync"] = weight_sync(args.model, args.tp, out, prompt_ids[2], checks)
    except Exception as error:  # recorded, so the merge checks still run
        summary["weight_sync_error"] = repr(error)
        checks.setdefault("weight_sync", False)

    # 5. merge every complete profile both ways
    checks["some_profile_complete"] = bool(profiled)
    for variant in profiled:
        profile_dir = out / f"profile_{variant}"
        ours = subprocess.run(
            [
                sys.executable,
                "-m",
                "rl_trace_observer.merger.cli",
                str(profile_dir),
                "-o",
                str(out / f"e0_{variant}_merged.json"),
            ],
            capture_output=True,
            text=True,
        )
        summary[variant]["rl_trace_merge"] = {"returncode": ours.returncode, "log": ours.stderr[-4000:]}
        checks[f"{variant}_rl_trace_merge"] = ours.returncode == 0
        ranks = []
        for rank in range(args.tp):
            ranks += [
                "--rank",
                str(rank),
                str(profile_dir / f"e0-TP{rank}.viztracer.json"),
                str(profile_dir / f"e0-TP{rank}.proton.chrome_trace"),
            ]
        official = subprocess.run(
            [
                sys.executable,
                "-m",
                "tokenspeed.cli",
                "merge-traces",
                "--all-ranks",
                *ranks,
                "-o",
                str(out / f"e0_{variant}_official.json"),
            ],
            capture_output=True,
            text=True,
        )
        summary[variant]["official_merge"] = {
            "returncode": official.returncode,
            "log": (official.stdout + official.stderr)[-4000:],
        }
        checks[f"{variant}_official_merge"] = official.returncode == 0

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(checks, indent=2))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
