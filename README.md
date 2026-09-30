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
