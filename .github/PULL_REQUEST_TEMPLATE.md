### What does this PR do?

> Add **concise** overview of what this PR aims to achieve or accomplish. Reference related GitHub issues and PRs that help with the review, including the work plan item in the [tracking issue](https://github.com/Luosuu/rl-trace-observer/issues/1).

### Checklist Before Starting

- [ ] Search for similar PRs. Paste at least one query link here: ...
- [ ] Format the PR title as `[{modules}] {type}: {description}` (This will be checked by the CI)
  - `{modules}` include `rl-insight`, `verl`, `tokenspeed`, `viztracer`, `torch`, `ray`, `merger`, `manifest`, `clock`, `deps`, `ci`, `doc`, `misc`
  - If this PR involves multiple modules, separate them with `,` like `[rl-insight, verl]`
  - `{type}` is in `feat`, `fix`, `refactor`, `chore`, `test`
  - If this PR breaks any API (environment variables, runtime env, config, function signature, trace output format, etc.), add `[BREAKING]` to the beginning of the title.
  - Example: `[BREAKING][merger, manifest] feat: read artifacts from the session manifest`

### Test

> For changes that can not be tested by CI (e.g., multi-node or GPU-only paths such as real rollout servers, Proton, or Torch CUDA traces), validate by experiment(s) and show results like Perfetto screenshots, trace excerpts, or overhead measurements.

### API and Usage Example

> Demonstrate how the API changes if any, and provide usage example(s) if possible.

```bash
# Add a command, environment variable, or ray_runtime_env.yaml snippet demonstrating how to use this
```

### Design & Code Changes

> Demonstrate the high-level design if this PR is complex, and list the specific changes. Reference the relevant section of [the design doc](https://github.com/Luosuu/rl-trace-observer/blob/main/docs/design.md).

### Checklist Before Submitting

> [!IMPORTANT]
> Please check all the following items before requesting a review, otherwise the reviewer might deprioritize this PR for review.

- [ ] Run lint and format checks: `uv run ruff check . && uv run ruff format --check .`
- [ ] Run the tests with the affected extras synced, e.g. `uv sync --all-extras && uv run pytest`.
- [ ] If dependencies changed, update the lock with `uv lock` and keep the extras in [`ray_runtime_env.yaml`](https://github.com/Luosuu/rl-trace-observer/blob/main/ray_runtime_env.yaml) consistent with `pyproject.toml`.
- [ ] Add / Update the documentation ([README](https://github.com/Luosuu/rl-trace-observer/blob/main/README.md) and [design doc](https://github.com/Luosuu/rl-trace-observer/blob/main/docs/design.md)).
- [ ] Add unit or end-to-end test(s) to [the CI workflow](https://github.com/Luosuu/rl-trace-observer/tree/main/.github/workflows) to cover all the code. If not feasible, explain why: ...
- [ ] Update the work plan in the [tracking issue](https://github.com/Luosuu/rl-trace-observer/issues/1).
