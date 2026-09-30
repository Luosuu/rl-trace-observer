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

The project is managed with [uv](https://docs.astral.sh/uv/). `uv.lock` pins
every dependency, including RL-Insight from the
[Luosuu/rl-insight](https://github.com/Luosuu/rl-insight/tree/tianle/server-backend-env)
fork that adds `RL_INSIGHT_SERVER_BACKEND` until it lands upstream.

```bash
uv sync --all-extras      # everything: verl, tokenspeed, rl-insight, viztracer
uv sync --extra rl-insight --extra viztracer   # or pick extras
uv run pytest -q
```

| Extra | Installs |
|---|---|
| `rl-insight` | RL-Insight (fork, pinned by commit) |
| `verl` | VERL >= 0.9.1 |
| `tokenspeed` | TokenSpeed |
| `viztracer` | VizTracer for the optional actor call stacks |

`pytest` and `ruff` come from the default `dev` group. tokenspeed 0.1.0 pins
`transformers==5.12.0` while verl 0.9.1 declares `transformers<5.11`; uv
overrides the latter so both install together.

With the `rl-insight` and `verl` extras, CPU-only multi-process tests run
VERL's real Ray single controller: a driver, two actor workers and a stand-in
rollout server each load the plugin through VERL's plugin discovery and write
their own semantic artifact. No GPU is required.

## Zero-patch VERL integration

Install this package and `rl-insight` in the VERL driver and worker runtime:

```bash
uv sync --extra verl --extra rl-insight

# VERL >= 0.9 loads the package's `verl.plugins` entry point automatically in
# every process that imports verl. With VERL_USE_EXTERNAL_PLUGINS=none, or on
# VERL versions without plugin discovery, load it explicitly:
# export VERL_USE_EXTERNAL_MODULES=rl_trace_observer.integrations.verl.register
export RL_TRACE_OUTPUT_DIR=/path/to/profile-artifacts

# RL-Insight currently requires a non-empty URL during initialization. The
# local RL Trace Observer backend does not contact this URL.
export RL_INSIGHT_SERVER_URL=local://rl-trace-observer
```

Select the custom monitor backend and enable the existing VERL logger. When
launching through `uv run`, Ray starts every worker with the same `uv run`
flags, so pass the extras there too:

```bash
uv run --extra verl --extra rl-insight python -m verl.trainer.main_ppo \
  trainer.logger='["console","rl_insight"]' \
  +trainer.rl_insight.server.backend=rl_trace_observer \
  ...
```

VERL workers initialize RL-Insight lazily without the trainer config, so they
would otherwise select RL-Insight's default `ray` backend. Loading the external
module routes them to the local JSONL client as well, so driver and worker spans
are both captured:

- If RL-Insight supports `RL_INSIGHT_SERVER_BACKEND`, the module sets it to
  `rl_trace_observer` in each process unless it is already set.
- Otherwise it replaces RL-Insight's default `ray` factory with the same client.

Set `RL_TRACE_CAPTURE_DEFAULT_BACKEND=0` to leave the default backend untouched.

Each process incrementally writes:

```text
<RL_TRACE_OUTPUT_DIR>/rl-insight-<hostname>-pid-<pid>.chrome.jsonl
```

Incremental JSONL output avoids relying on graceful Ray worker shutdown. The
final merger can combine these absolute-time state events with actor Torch
Profiler traces and TokenSpeed VizTracer/Proton traces.

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
