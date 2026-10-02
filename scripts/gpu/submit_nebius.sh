#!/usr/bin/env bash
# Submit scripts/gpu/run_e2e.sh as a Nebius AI job on one 8xH100 node.
#
#   BUCKET_ID=storagebucket-... SUBNET_ID=vpcsubnet-... scripts/gpu/submit_nebius.sh [COMMIT]
#
# COMMIT (default: HEAD) must be pushed. Results: s3://<bucket>/rl-trace-observer/<job name>/.
# Extra variables for run_e2e.sh (STEPS, N_GPUS, ROLLOUT_TP, RUN_E0, ...) are passed through EXTRA_ENV,
# e.g. EXTRA_ENV="--env RUN_E0=0".
set -euo pipefail
COMMIT=${1:-$(git rev-parse HEAD)}
NAME=${NAME:-rl-trace-e2e-$(date +%Y%m%d-%H%M%S)-${COMMIT:0:7}}
nebius ai job create \
  --name "$NAME" \
  --image "${IMAGE:-nvidia/cuda:13.0.2-devel-ubuntu24.04}" \
  --platform "${PLATFORM:-gpu-h100-sxm}" \
  --preset "${PRESET:-8gpu-128vcpu-1600gb}" \
  --disk-size "${DISK_SIZE:-500Gi}" \
  --shm-size 64Gi \
  --timeout "${TIMEOUT:-6h}" \
  --subnet-id "$SUBNET_ID" \
  --volume "$BUCKET_ID:/mnt/results:rw" \
  --inject-file "$(dirname "$0")/nebius_job.sh:/opt/rl-trace/nebius_job.sh" \
  --env COMMIT="$COMMIT" --env JOB_NAME="$NAME" --env RESULTS_ROOT=/mnt/results/rl-trace-observer \
  ${EXTRA_ENV:-} \
  --container-command /bin/bash --args /opt/rl-trace/nebius_job.sh \
  --async
echo "submitted $NAME"
