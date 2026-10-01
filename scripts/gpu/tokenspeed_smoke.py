"""E0: TokenSpeed on its own, through the HTTP API the VERL rollout uses.

Launches ``tokenspeed serve`` (TP=2 by default), then checks and records:

1. ``/generate`` with ``input_ids`` returns token ids with one log prob each;
2. a ``VIZTRACER`` + ``PROTON`` profile over concurrent requests writes one
   VizTracer report and one Proton Chrome trace per TP rank, each with its
   ``baseTimeNanoseconds`` anchor;
3. releasing and resuming weights and KV cache keeps generation working;
4. ``rl-trace-merge`` and ``tokenspeed merge-traces --all-ranks`` both merge
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
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


class Server:
    def __init__(self, model: str, tp: int, log: Path):
        self.port, self.control_port = _free_port(), _free_port()
        env = {
            **os.environ,
            "TOKENSPEED_KERNEL_PROFILE_DATA": "trace",
            "TOKENSPEED_KERNEL_PROFILE_OUTPUT_FORMAT": "chrome_trace",
        }
        command = [
            sys.executable, "-m", "tokenspeed.cli", "serve",
            "--model", model, "--tp", str(tp),
            "--host", "127.0.0.1", "--port", str(self.port), "--control-port", str(self.control_port),
            "--gpu-memory-utilization", "0.5", "--enable-output-logprobs", "--enable-memory-saver",
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

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
        self.log.close()


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
        tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True)
        for p in prompts
    ]

    server = Server(args.model, args.tp, out / "tokenspeed_serve.log")

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
        summary["generate_example"] = output
        token_ids = output.get("output_ids") or []
        logprobs = (output.get("meta_info") or {}).get("output_token_logprobs") or []
        checks["generate_logprobs"] = bool(token_ids) and len(token_ids) == len(logprobs)
        summary["generate_text"] = tokenizer.decode(token_ids)

        # 2. profile concurrent requests
        profile_dir = out / "profile"
        start = time.time()
        summary["start_profile"] = server.post(
            "/start_profile",
            {"output_dir": str(profile_dir), "activities": ["VIZTRACER", "PROTON"], "profile_id": "e0"},
        )
        with concurrent.futures.ThreadPoolExecutor(16) as pool:
            outputs = list(pool.map(generate, prompt_ids))
        summary["stop_profile"] = server.post("/stop_profile")
        summary["profile_seconds"] = time.time() - start
        checks["profiled_requests"] = all(o.get("output_ids") for o in outputs)
        deadline = time.monotonic() + 120
        expected = {
            f"e0-TP{rank}.{suffix}" for rank in range(args.tp) for suffix in ("viztracer.json", "proton.chrome_trace")
        }
        while not expected <= {p.name for p in profile_dir.glob("*")} and time.monotonic() < deadline:
            time.sleep(1)
        files = sorted(p.name for p in profile_dir.glob("*"))
        summary["profile_files"] = {name: (profile_dir / name).stat().st_size for name in files}
        checks["profile_files"] = expected <= set(files)
        anchors = {}
        for name in sorted(expected & set(files)):
            data = json.loads((profile_dir / name).read_text())
            anchor = data.get("viztracer_metadata", {}).get("baseTimeNanoseconds") or data.get("baseTimeNanoseconds")
            flows = sum(1 for e in data.get("traceEvents", []) if e.get("ph") in ("s", "f"))
            scopes = sum(
                1 for e in data.get("traceEvents", []) if isinstance(e.get("args"), dict) and "scope_id" in e["args"]
            )
            anchors[name] = {
                "baseTimeNanoseconds": anchor,
                "events": len(data.get("traceEvents", [])),
                "flows": flows,
                "scope_ids": scopes,
            }
        summary["profile_anchors"] = anchors
        checks["profile_anchors"] = bool(anchors) and all(a["baseTimeNanoseconds"] for a in anchors.values())

        # 3. sleep and wake
        for tags in (["kv_cache", "weights"],):
            summary["release"] = server.post("/release_memory_occupation", {"tags": tags})
            summary["resume"] = server.post("/resume_memory_occupation", {"tags": tags})
        checks["generate_after_wake"] = bool(generate(prompt_ids[1]).get("output_ids"))
    finally:
        server.close()

    # 4. merge
    ours = subprocess.run(
        [sys.executable, "-m", "rl_trace_observer.merger.cli", str(out / "profile"), "-o", str(out / "e0_merged.json")],
        capture_output=True,
        text=True,
    )
    summary["rl_trace_merge"] = {"returncode": ours.returncode, "log": ours.stderr[-4000:]}
    checks["rl_trace_merge"] = ours.returncode == 0
    ranks = []
    for rank in range(args.tp):
        ranks += [
            "--rank",
            str(rank),
            str(out / f"profile/e0-TP{rank}.viztracer.json"),
            str(out / f"profile/e0-TP{rank}.proton.chrome_trace"),
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
            str(out / "e0_official.json"),
        ],
        capture_output=True,
        text=True,
    )
    summary["official_merge"] = {"returncode": official.returncode, "log": (official.stdout + official.stderr)[-4000:]}
    checks["official_merge"] = official.returncode == 0

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(checks, indent=2))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
