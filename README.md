# RL Trace Observer

> **Status:** component proof of concept. The end-to-end session merger and
> distributed artifact collection are still under development.

See [Global RL Trace Bridge Design](docs/design.md) for the project goals,
architecture, current limitations, and work plan.

An external trace integration for VERL and TokenSpeed. The default integration
reuses VERL's existing RL-Insight state annotations and does not patch VERL.

Existing annotations include actor training stages such as `train_batch`,
`actor_compute_log_prob`, and `actor_update`, plus rollout stages such as
`vllm_generate` and `sglang_generate`. RL Trace Observer registers an
RL-Insight monitor client that converts those spans into append-only Chrome
Trace JSONL files.

An optional, reversible `DistProfiler.start/stop` patch is retained only when
actor-side VizTracer call stacks are needed.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/) on Python 3.12.
`uv.lock` pins every dependency: VERL (>= 0.9.1), VizTracer, and RL-Insight
from the
[Luosuu/rl-insight](https://github.com/Luosuu/rl-insight/tree/tianle/server-backend-env)
fork that adds `RL_INSIGHT_SERVER_BACKEND` until it lands upstream. TokenSpeed
is the only extra.

```bash
uv sync --extra tokenspeed
uv run pytest -q
```

`pytest` and `ruff` come from the default `dev` group. tokenspeed 0.1.0 pins
`transformers==5.12.0` while verl 0.9.1 declares `transformers<5.11`; uv
overrides the latter so both install together.

The tests run on CPU, including multi-process tests on VERL's real Ray single
controller: a driver, two actor workers and a stand-in rollout server each load
the plugin through VERL's plugin discovery and write their own semantic
artifact. No GPU is required.

### CPU PPO end-to-end test

`tests/test_cpu_ppo.py` runs VERL's real `verl.trainer.main_ppo` for one PPO
step on CPU (about 90 s): a driver, two FSDP2 actor workers with the Torch
profiler enabled, and two rollout replicas. It needs no GPU or network access:

- `tests/cpu_ppo/assets.py` builds a tiny Qwen2 model, a byte-level BPE ChatML
  tokenizer and a parquet dataset offline.
- `tests/cpu_ppo/cpu_plugin.py`, loaded through `VERL_USE_EXTERNAL_MODULES`,
  registers a `cpu` platform, the FSDP engines for cpu and a `mock` rollout
  whose servers return short random completions, using only VERL's extension
  points (plus one workaround listed in its docstring).

The test merges the resulting RL-Insight and Torch artifacts with
`rl-trace-merge` and checks in Perfetto that every rank's `actor_update` lines
up on both timelines.

## Zero-patch VERL integration

VERL >= 0.9 loads the package's `verl.plugins` entry point automatically in
every process that imports verl. With `VERL_USE_EXTERNAL_PLUGINS=none`, load it
explicitly with
`VERL_USE_EXTERNAL_MODULES=rl_trace_observer.integrations.verl.register`.

Run on Ray with [`ray_runtime_env.yaml`](ray_runtime_env.yaml), following Ray's
uv integration: the project directory is uploaded as `working_dir` and every
worker starts through `uv run --locked --extra tokenspeed python`, so all
processes use the environment pinned by `uv.lock`. Set `RL_TRACE_OUTPUT_DIR` in
the yaml to an absolute directory shared by all nodes; it is required, and a
relative path is rejected because each worker runs inside a temporary copy of
the working_dir. Then submit with the same extras on the driver:

```bash
ray job submit --runtime-env ray_runtime_env.yaml -- \
  uv run --locked --extra tokenspeed python -m verl.trainer.main_ppo \
  trainer.logger='["console","rl_insight"]' \
  ...
```

`trainer.logger` containing `rl_insight` makes VERL enable RL-Insight in every
worker. VERL workers initialize RL-Insight lazily without the trainer config, so
the plugin selects its backend through `RL_INSIGHT_SERVER_BACKEND`, which it
defaults to `rl_trace_observer` in each process; an explicitly set value wins.
This requires the pinned RL-Insight fork; the plugin raises an error on an
RL-Insight without `RL_INSIGHT_SERVER_BACKEND`.

Each process incrementally writes:

```text
<RL_TRACE_OUTPUT_DIR>/rl-insight-<hostname>-pid-<pid>.chrome.jsonl
```

Incremental JSONL output avoids relying on graceful Ray worker shutdown. The
final merger can combine these absolute-time state events with actor Torch
Profiler traces and TokenSpeed VizTracer/Proton traces.

## Merge into one Perfetto trace

`rl-trace-merge` combines the artifacts of a profiling run into a single Chrome
Trace that opens in [Perfetto](https://ui.perfetto.dev):

```bash
uv run rl-trace-merge /shared/profile-artifacts /path/to/torch/save_path -o merged.json
```

Inputs are files or directories, searched recursively for:

| Artifact | Written by | Time anchor |
|---|---|---|
| `rl-insight-<host>-pid-<pid>.chrome.jsonl` | the RL-Insight backend above | epoch µs `ts` |
| `<role>_..._rank<r>_pid<pid>_<timestamp>.json[.gz]` | VERL `global_profiler.tool=torch` | `baseTimeNanoseconds` |
| `step-<s>-role-<r>-rank-<n>-pid-<pid>.viztracer.json` | the optional actor VizTracer | `viztracer_metadata.baseTimeNanoseconds` |
| `rl-trace-process-<host>-pid-<pid>.json` | each tracing process (process record) | clock snapshot only |

Every process that writes an artifact also writes a process record with its
hostname, pid, run id, role, Ray job/node/actor identity, `torch.distributed`
rank, the versions of the profilers and frameworks it loaded, a wall/monotonic
clock snapshot and the files it registered. The merger uses the records to link
Torch and VizTracer traces, whose filenames carry only a pid, to their host: by
pid, and by rank when several hosts reused a pid.

### Runs and profiled steps

Every process names the run it belongs to. By default the run id is the Ray
session and job (`session_…-job-01000000`), which all processes of one training
job share; set `RL_TRACE_RUN_ID` to name runs yourself (every process must see
it, e.g. through `ray_runtime_env.yaml`'s `env_vars`). The trainer also records
a `global_step` span around each step of VERL's v1 trainer (`PPOTrainer.step`)
on its `trainer` lane, and the manifest lists every profiled step as a session
(`<run_id>-step-<n>`) with its time window and its Torch and VizTracer traces.

```bash
uv run rl-trace-merge "$RL_TRACE_OUTPUT_DIR" -o step3.json --step 3        # one step
uv run rl-trace-merge "$RL_TRACE_OUTPUT_DIR" -o run.json --run <run_id>     # one of several runs
```

`--step` keeps that step's Torch and VizTracer traces and the RL-Insight spans
that overlap its window; repeat it for several steps. Windows are measured on
the trainer's clock, and in VERL's asynchronous trainers generation for the next
step can overlap the current one.

Every source is placed on one timeline starting at the earliest event. All
sources of one OS process share one Perfetto process named
`<hostname> pid <pid> · <Ray actor name>`, with thread names prefixed by their
source (`RL-Insight · rank_0`, `Torch · thread …`, `VizTracer · …`). Other pids
in a source, such as Kineto GPU devices, get their own process. Threads get
trace-wide unique tids and flow ids are renumbered per source, so identical OS
pids, tids or Kineto flow ids on different nodes never collide.

Next to the trace, `rl-trace-merge` writes the session manifest
(`<output>.manifest.json`, or `--manifest PATH`): every process, every artifact
with its size, sha256, process and completeness, and the problems found:

| Problem | Meaning |
|---|---|
| `incomplete` | an artifact was cut short (truncated JSONL line), cannot be read or is malformed; or a process record has an unsupported `schema_version` |
| `duplicate` | a non-empty artifact is a copy of another input (same file name and content), and is merged once |
| `missing` | a process registered an artifact that is not in the inputs |
| `unlinked` | no process record matches a Torch or VizTracer trace's pid (and, when both are known, rank) |
| `ambiguous` | several process records match and the rank cannot tell them apart |
| `mixed_runs` | the inputs hold artifacts of several runs and `--run` chose none |
| `no_step_window` | a `--step` has no `global_step` span, so RL-Insight spans are left out |

Problems and dropped events (negative durations) are warnings; `--strict` turns
them into a failure. The manifest is written even when no trace is (nothing
readable, no timed events, or a `--strict` failure), and a trace left at the
output path by an earlier run is then removed. The trace and the manifest are
each replaced atomically, and neither may be one of the inputs.

Timestamps are each host's wall clock, so cross-node ordering is only as good
as the nodes' clock synchronization (NTP/PTP).

## Preserve the normal RL-Insight backend

Set the following to make the custom client forward all events to the existing
Ray monitor client after writing local trace spans:

```bash
export RL_TRACE_FORWARD_TO_RL_INSIGHT=1
export RL_INSIGHT_SERVER_URL=http://<rl-insight-server>:18080
```

The configured backend remains `rl_trace_observer`; it creates RL-Insight's
original Ray client internally as its delegate. Forwarding failures are logged
and never interrupt training.

## Optional actor VizTracer

Actor-side VizTracer still needs lifecycle control. Enable the thin, reversible
monkey patch explicitly:

```bash
export RL_TRACE_VIZTRACER=1
export RL_TRACE_VIZTRACER_MIN_DURATION_US=100
export RL_TRACE_VIZTRACER_ROLES=e2e
```

With `RL_TRACE_VIZTRACER` unset, importing the external module does not modify
`DistProfiler`.
