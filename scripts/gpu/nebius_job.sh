#!/usr/bin/env bash
# Entry point of a Nebius AI job: check out $COMMIT and run scripts/gpu/run_e2e.sh,
# writing results under the object-storage mount $RESULTS_ROOT/$JOB_NAME.
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git rsync curl ca-certificates build-essential >/dev/null
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
git clone -q https://github.com/Luosuu/rl-trace-observer /root/rl-trace-observer
cd /root/rl-trace-observer && git checkout -q "$COMMIT" || exit 1
RESULTS_DIR="$RESULTS_ROOT/$JOB_NAME" bash scripts/gpu/run_e2e.sh
