# 2026-10-02：one-step-off-policy 下的 TokenSpeed rollout

同步 trainer 的 hybrid 部署里，TokenSpeed 和 actor 共用 4 张 GPU，生成和训练只能轮流进行（[2026-10-02 CUDA graph 记录](2026-10-02-tokenspeed-cuda-graph-proton.md)）。本次换成 VERL 0.9.1 的 one-step-off-policy trainer：actor 用 GPU 0–3，2 个 standalone TokenSpeed replica（TP=2）用 GPU 4–7。第 n 步训练第 n 批数据时，replica 同时生成第 n+1 批。

复现方式：`TRAINER=one_step_off RUN_E0=0 PLATFORM=gpu-h200-sxm scripts/gpu/submit_nebius.sh`（commit `e8eb599`，8×H200）。

## 做法

- **standalone replica**：`TokenSpeedReplica` 支持 VERL 的 standalone 模式，在该 replica 的 checkpoint-engine worker 所在 GPU 上启动 `tokenspeed serve`。
- **权重同步：`checkpoint_engine.backend=tokenspeed`**：VERL 现有的 backend 先把权重送到每张 rollout GPU 上的 checkpoint-engine worker，再经 CUDA IPC 交给推理 server，而 TokenSpeed 没有 IPC 接收路径。这个 backend 改为：
  - 训练 rank 0 与所有 TokenSpeed rank 建一个权重组（1 + 4 个 rank），每个 bucket 只广播一次；
  - 其他训练 rank 只参与 full tensor 的 gather；
  - rollout 一侧的 worker 什么也不做。
  - 权重组在第一次同步时建立，之后一直复用。hybrid 模式下“至少 2 个 replica、由别的 replica 的训练 rank 发送”的限制，在这里不存在。
- **profile 与 step 窗口**：VERL 的这个 trainer 不对 rollout 做 profile，也没有我们的 `global_step` 标记。插件补上两件事：
  1. 为每一步记录 `global_step` 窗口；
  2. 在 profile 的第 n 步里，对同时进行的那次生成（第 n+1 批）做 profile，并登记为第 n 步。这样第 n 步的 trace 里，actor 训练和 rollout 并排出现。
- **启动 profile 的时机**：driver 的 event loop 在训练阶段之间只在 `await asyncio.sleep(0)` 处让出。因此生成任务在派发请求之前，用阻塞调用启动 profile，以免生成被推迟到下一个训练阶段结束之后。

## 结果

5 步全部完成。每一步（含第 0 步的初始同步）都有一行 `sent 291 tensors to TokenSpeed replicas [0, 1]`。`rl-trace-merge --strict --step 3` 和 `--step 4` 退出码均为 0，manifest 无任何问题。每个 profile 的 step 都包含：

- trainer 的 RL-Insight；
- 4 个 actor rank 的 RL-Insight 和 Torch；
- 2 个 replica 共 4 个 TP rank 的 VizTracer 和 Proton。

| step | step 时长 (s) | 等待数据 (s) | 这批数据的生成 (s) | actor 训练 (s) | 权重同步 (s) | profile |
|---|---|---|---|---|---|---|
| 1 | 20.8 | 10.3 | 10.1 | 10.4 | 0.17 | |
| 2 | 3.2 | 0.0 | 10.4 | 3.0 | 0.16 | |
| 3 | 4.9 | 0.0 | 3.0 | 4.6 | 0.21 | ✓ |
| 4 | 13.3 | 5.4 | 4.7 | 6.4 | 1.45 | ✓ |
| 5 | 9.1 | 5.7 | 6.5 | 3.2 | 0.24 | |

“等待数据”是 VERL 的 `timing_s/gen`：这一步开始时，等上一步发起的生成完成所用的时间。“这批数据的生成”是 `timing_s/generate_async`。

- **第 1 步**：没有可以并行的上一步，所以要等第一批生成完成。
- **第 2、3 步**：等待为 0，生成完全藏在训练后面。第 1 步里第 2 批的生成（10.4 s）包含首次请求的启动开销。
- **第 4、5 步**：要等约 5.5 s。它们的数据分别是在第 3、4 步（profile 的步）里生成的，生成任务在结束 profile 时要等 TokenSpeed 写完 VizTracer 和 Proton 文件（每个 rank 约 60–80 MB），所以晚于训练完成。这部分是 profile 的开销，不是重叠失败。
- **同步 trainer 对照**：在同样的 H200 上，hybrid 同步 trainer 不开 profile 的步骤约 8.6–12.7 s，one-step-off 是 3.2–4.9 s。不过后者多用了 4 张 GPU，两者不是等资源的比较。
- **训练变成 off-policy**：VERL 的 one-step-off 配置默认 `rollout_correction.bypass_mode=True`，直接用 rollout 的 logprob 作为 old logprob，所以没有 `rollout_probs_diff` 指标。reward 也不能和同步版直接比。

### RL-Insight 上的重叠

在每个 `global_step` 窗口里，比较 `tokenspeed_generate` 请求的时间范围和 `actor_update` 的时间范围（时间相对窗口起点）：

| step | 窗口 (s) | 生成（下一批） | actor_update | 重叠 (s) |
|---|---|---|---|---|
| 2 | 3.2 | 0.2–2.0 s，128 个请求 | 0.3–3.2 s | 1.7 |
| 3 | 27.9 | 0.3–2.6 s，128 个请求 | 0.7–4.9 s | 1.9 |
| 4 | 35.8 | 6.9–9.1 s，128 个请求 | 6.9–13.3 s | 2.2 |

下一批的生成几乎全部落在本步 actor 训练期间。第 3、4 步的窗口比 step 时长长出 20 多秒，原因是 VERL 在 step 计时之外停止 actor 的 Torch profiler 并导出 trace，而我们的 `global_step` 窗口覆盖整个 `fit_step`。

## 后台保存 profile（commit `9eded46`，8×H100）

上面第 4、5 步的等待，以及第 3、4 步窗口比 step 长出的 20 多秒，都来自保存 profile。现在所有保存都在后台进行：

- **TokenSpeed**：
  - `/stop_profile` 只停止记录，VizTracer 和 Proton 的文件改由后台线程写出；
  - server actor 在后台等文件写完再登记，检查方式只读文件末尾，不再完整解析 JSON；
  - 下一次 `/start_profile` 会先等上一次写完。
- **driver**：不再等 rollout 的 `stop_profile`。
- **actor**：Torch trace 在后台线程导出。如果配置了 VERL 的 finish hook，它会先等导出完成。
- **训练结束**：`fit` 结束前等待所有进程把文件写完并登记，以免 Ray 回收进程时丢文件。

配置同上，机器换成 H100（H200 当时没有容量）：

| step | step 时长 (s) | 等待数据 (s) | actor 训练 (s) | `stop_profile` (s) | `global_step` 窗口 (s) | profile |
|---|---|---|---|---|---|---|
| 1 | 21.0 | 10.5 | 10.3 | 0 | 21.0 | |
| 2 | 4.3 | 0.0 | 4.1 | 0 | 4.3 | |
| 3 | 4.8 | 0.0 | 4.4 | 6.0 | 10.8 | ✓ |
| 4 | 6.6 | 0.0 | 6.2 | 6.1 | 12.7 | ✓ |
| 5 | 8.0 | 0.0 | 7.6 | 0 | 8.0 | |

- **等待数据**：第 2–5 步都是 0，profile 的步不再拖慢下一步（之前第 4、5 步各等 5.4 s、5.7 s）。
- **profile 步的窗口**：从 27.9 s / 35.8 s 降到 10.8 s / 12.7 s。剩下约 6 s 是 VERL 在 actor 上调用 `stop_profile`，导出已经移到后台，这部分应是 Torch profiler 停止时收集和整理 CUDA activity 的时间。它只影响 actor 和等待它的 driver，rollout 的生成不受影响。
- **重叠**：第 3、4 步中，下一批的 128 个请求在 0.4–2.7 s 内完成，都落在 `actor_update` 期间。
- **完整性**：第 3、4 步的 `--strict` 合并都成功，manifest 无问题。每一步都有全部 19 个产物：7 个 RL-Insight、4 个 Torch、4 个 Proton、4 个 VizTracer。

产物：`s3://bench-artifacts-e00wpqevpr0037mwj6dvm8/rl-trace-observer/rl-trace-asyncsave-20261002-164929-9eded46/`。

## 产物

`s3://bench-artifacts-e00wpqevpr0037mwj6dvm8/rl-trace-observer/rl-trace-onestep-20261002-151244-e8eb599/`：`e2/step3.json.gz`、`e2/step4.json.gz`、`e2/artifacts/`、`e2/metrics.txt`。

## 尚未完成

- fully-async trainer（rollouter 与 trainer 通过消息队列解耦、partial rollout）。
- actor 上 Torch profiler 停止时约 6 s 的收集时间仍在 step 内（VERL 的 profiler 实现）。
- 等资源对照：例如同步 trainer 用 8 卡 hybrid，对比 one-step-off 的 4+4。
