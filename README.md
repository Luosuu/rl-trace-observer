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

## Local validation

```bash
uv venv --python 3.12
uv pip install -e '.[test,viztracer]'
uv run --no-sync pytest -q
```

## Zero-patch VERL integration

Install this package and `rl-insight` in the VERL driver and worker runtime:

```bash
uv pip install -e '.[rl-insight]'

export VERL_USE_EXTERNAL_MODULES=rl_trace_observer.integrations.verl.register
export RL_TRACE_OUTPUT_DIR=/path/to/profile-artifacts

# RL-Insight currently requires a non-empty URL during initialization. The
# local RL Trace Observer backend does not contact this URL.
export RL_INSIGHT_SERVER_URL=local://rl-trace-observer
```

Select the custom monitor backend and enable the existing VERL logger:

```bash
python -m verl.trainer.main_ppo \
  trainer.logger='["console","rl_insight"]' \
  +trainer.rl_insight.server.backend=rl_trace_observer \
  ...
```

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

The configured backend remains `rl_trace_observer`; it creates the normal Ray
client internally as its delegate.

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
