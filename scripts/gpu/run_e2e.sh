#!/usr/bin/env bash
# GPU end-to-end run (E0 + E2 of Luosuu/rl-trace-observer#10) on one 8-GPU node:
#
#   E0  TokenSpeed alone (TP=2): generation, VIZTRACER+PROTON profile, sleep/wake, merge.
#   E2  VERL GRPO on GSM8K with rollout.name=tokenspeed: Qwen2.5-0.5B-Instruct,
#       FSDP2 actor on 4 GPUs, 2 hybrid TokenSpeed replicas x TP=2, 5 steps,
#       steps 3 and 4 profiled (actor Torch cpu+cuda, TokenSpeed VizTracer+Proton,
#       RL-Insight), then `rl-trace-merge --strict --step N`.
#
# Run from the repository root. Results go to $RESULTS_DIR (default ./gpu-results/<time>);
# everything is written to local disk and copied there once at the end, since
# object-storage mounts cannot overwrite or rename files.
set -uo pipefail

REPO_DIR=$(pwd)
RESULTS_DIR=${RESULTS_DIR:-$REPO_DIR/gpu-results/$(date +%Y%m%d-%H%M%S)}
WORK_DIR=${WORK_DIR:-/tmp/rl-trace-e2e}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
STEPS=${STEPS:-5}
PROFILE_STEPS=${PROFILE_STEPS:-[3,4]}
N_GPUS=${N_GPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
RUN_E0=${RUN_E0:-1}
RUN_E2=${RUN_E2:-1}
mkdir -p "$RESULTS_DIR" "$WORK_DIR"
mkdir -p "$WORK_DIR/out"
exec > >(tee -a "$WORK_DIR/out/job.log") 2>&1

finish() {
  echo "=== copying results to $RESULTS_DIR"
  cp -r "$WORK_DIR/out/." "$RESULTS_DIR/" || true
}
trap finish EXIT

echo "=== $(date -u) commit $(git rev-parse HEAD 2>/dev/null)"
nvidia-smi || true

echo "=== environment"
export UV_PROJECT_ENVIRONMENT=$WORK_DIR/venv UV_HTTP_TIMEOUT=600
command -v uv >/dev/null || pip install -q uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv sync --locked --extra tokenspeed || exit 1
# JIT builds (flashinfer) run the venv's ninja.
export PATH=$UV_PROJECT_ENVIRONMENT/bin:$PATH
PY=$UV_PROJECT_ENVIRONMENT/bin/python
$PY -c "import torch, tokenspeed; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.device_count(), 'gpus')"
$PY -m tokenspeed.cli env > "$WORK_DIR/out/tokenspeed_env.txt" 2>&1 || true

MODEL_DIR=$WORK_DIR/models/$(basename "$MODEL")
[ -d "$MODEL_DIR" ] || $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', local_dir='$MODEL_DIR')" || exit 1
[ -f "$WORK_DIR/data/train.parquet" ] || $PY scripts/gpu/gsm8k.py "$WORK_DIR/data" || exit 1

if [ "$RUN_E0" = 1 ]; then
  echo "=== E0: TokenSpeed alone"
  NCCL_DEBUG=${E0_NCCL_DEBUG:-INFO} CUDA_VISIBLE_DEVICES=0,1,2 $PY scripts/gpu/tokenspeed_smoke.py --model "$MODEL_DIR" --out "$WORK_DIR/out/e0" --tp 2
  echo "E0 exit $?"
fi

[ "$RUN_E2" = 1 ] || exit 0
echo "=== E2: VERL GRPO with TokenSpeed rollout"
E2=$WORK_DIR/out/e2
mkdir -p "$E2"
export RL_TRACE_OUTPUT_DIR=$E2/artifacts
export RL_INSIGHT_SERVER_URL=local://rl-trace-observer
export VERL_RL_INSIGHT_ENABLE=1
export TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
start=$(date +%s)
$PY -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$WORK_DIR/data/train.parquet" \
  data.val_files="$WORK_DIR/data/test.parquet" \
  data.train_batch_size=32 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path="$MODEL_DIR" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.rollout.name=tokenspeed \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  global_profiler.tool=torch \
  "global_profiler.steps=$PROFILE_STEPS" \
  global_profiler.save_path="$E2/artifacts/torch" \
  actor_rollout_ref.actor.profiler.enable=True \
  actor_rollout_ref.actor.profiler.all_ranks=True \
  "actor_rollout_ref.actor.profiler.tool_config.torch.contents=[cuda,cpu]" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.total_training_steps="$STEPS" \
  'trainer.logger=["console","rl_insight"]' \
  trainer.val_before_train=False \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.project_name=rl_trace_observer \
  trainer.experiment_name=tokenspeed_e2e \
  hydra.run.dir="$E2/hydra" \
  2>&1 | tee "$E2/main_ppo.log" | grep --line-buffered -E "step:[0-9]+ |Traceback|Error|Sent [0-9]+ tensors|TokenSpeed ready"
status=${PIPESTATUS[0]}
echo "E2 main_ppo exit $status after $(( $(date +%s) - start ))s"
grep -E "step:[0-9]+ " "$E2/main_ppo.log" | tail -n "$STEPS" > "$E2/metrics.txt" || true
tar czf "$E2/ray_logs.tgz" -C /tmp/ray/session_latest logs 2>/dev/null || true

echo "=== merge"
for step in $(echo "$PROFILE_STEPS" | tr -d '[] ' | tr , ' '); do
  $PY -m rl_trace_observer.merger.cli "$RL_TRACE_OUTPUT_DIR" -o "$E2/step$step.json" --strict --step "$step"
  echo "merge step $step exit $?"
done
$PY -m rl_trace_observer.merger.cli "$RL_TRACE_OUTPUT_DIR" -o "$E2/all_steps.json"
exit $status
