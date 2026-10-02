# 2026-10-02：CUDA graph 下的 TokenSpeed Proton profile

[2026-10-01 的记录](2026-10-01-tokenspeed-grpo.md)里，Proton 在 CUDA graph 下 finalize 失败，因此 rollout 用 `enforce_eager=True` 运行。本次改为常驻 Proton session（`integrations/tokenspeed/proton_graphs.py`），在 CUDA graph 下重跑 E0 和 E2。

复现方式：`PLATFORM=gpu-h200-sxm scripts/gpu/submit_nebius.sh`（commit `93518a4`，`ROLLOUT_EAGER=False`，现在是默认值）。

## 原因（triton 3.8 / tokenspeed-proton 3.8.10 源码）

- **capture 时的记录**：Proton 在 CUPTI 的 graph-node 创建回调里，为每个已注册的 session 记下节点的 capture 调用栈（`CuptiProfiler.cpp`，`handleGraphResourceCallbacks`）。
- **回放时的归属**：kernel 能对应到节点，graph 回放的 kernel 就挂在 `<captured_at>` 帧下。如果 capture 时这个 session 还不存在，kernel 只能作为 graph launch op 的子节点。这个 launch op 没有 CPU 时间范围，于是 trace 模式在写出 `launch->kernel` flow 时抛出 "Cannot find CPU scope event for kernel launch"（`ChromeTrace.cpp`）。
- **TokenSpeed 的做法**：`tokenspeed serve` 在启动时（`EventLoop` 构造期间）capture graph，而每次 `/start_profile` 都调用 `proton.start` 新建 session，`/stop_profile` 又 `finalize` 结束它。所以它的 session 永远看不到 capture。

## 做法

不修改 TokenSpeed，仍然通过已有的 `sitecustomize` 注入：

- **建立 session**：在 `EventLoop.__init__` 之前，为每个 scheduler 建一个 Proton session（`data="trace"`、`hook="triton"`、`mode="periodic_flushing:format=chrome_trace"`），构造完成后停用。
- **开始 profile**：`/start_profile` 不再新建 session，而是调用 `advance_phase` 进入新的 data phase，再 `activate`。同时设置 `tokenspeed_kernel` 的 `ProfilingState`，让 `kernel_scope()` 照常记录 CPU scope 和 VizTracer flow。
- **结束 profile**：`/stop_profile` 时：
  1. 再调用一次 `advance_phase`；
  2. 在新的 phase 里发射一个 `torch.cuda._sleep(1)`。periodic flushing 只有在之后的 phase 出现 kernel 记录时，才会把之前的 phase 写出。这个调用不经过 torch dispatch，所以不会触发 TokenSpeed control-plane 线程的 `_NoDeviceWork` 检查；
  3. `deactivate(flushing=True)`，等 `<session>.part_<phase>.chrome_trace` 写完，再改名为 TokenSpeed 原本要写的 `<profile_id>-<rank>.proton.chrome_trace`；
  4. 两次 profile 之间的 phase 文件直接删除。

`enforce_eager=True` 时不注入上述逻辑，TokenSpeed 自己的 Proton 流程不变。

## E0：TokenSpeed 单独运行（TP=2，32 个并发请求）

| | eager | CUDA graph，第 1 次 profile | CUDA graph，第 2 次 profile |
|---|---|---|---|
| profile 时长（s） | 3.6 | 0.7 | 0.7 |
| 每个 rank 的 Proton kernel 事件 | 12826 | 13478 | 18543 |
| 其中带 `<captured_at>` 的（graph 回放） | — | 11688 | — |
| VizTracer→Proton scope flow | 6192 | 240 | 216 |
| `rl-trace-merge` / `tokenspeed merge-traces` | 通过 | 通过 | 通过 |

- **同一 server 两次 profile**：两次之间做了一次 release/resume，session 跨过两次 profile 都正常写出。
- **CPU scope 与 flow 变少**：graph 回放不经过 Python，所以只有 prefill 等 eager 路径还会产生 `kernel_scope`。
- **graph scope 事件**：Proton 按 capture 调用栈重建了 graph scope 事件（`cat=scope`，名为 `<captured_at>`）。
- **示例**：一个回放 kernel 的调用栈是 `ROOT → <captured_at> → kernel_cutlass_…RMSNormKernel…`。
- **其他检查**：server 日志中没有 Proton 警告，权重同步检查全部通过。

## E2：VERL GRPO 端到端（8×H200）

配置与 2026-10-01 相同，只改了 `enforce_eager=False`，机器从 H100 换成 H200（H100 当时没有容量）。5 步全部完成；`rl-trace-merge --strict --step 3` 和 `--step 4` 退出码均为 0，manifest 无任何问题。

| step | step 时长 (s) | 生成 (s) | 权重同步 (s) | `rollout_probs_diff_mean` | profile |
|---|---|---|---|---|---|
| 1 | 30.9 | 4.1 | 0.70 | 0.0044 | |
| 2 | 8.6 | 2.1 | 1.01 | 0.0046 | |
| 3 | 44.2 | 7.4 | 0.78 | 0.0041 | ✓ |
| 4 | 45.3 | 7.5 | 0.78 | 0.0044 | ✓ |
| 5 | 12.7 | 2.1 | 0.88 | 0.0045 | |

- **生成速度**：不开 profile 的生成从 eager 下的 12.1 s 降到 2.1 s，开 profile 时从约 37 s 降到 7.4 s。机器不同（H200 对 H100），所以只能看量级。
- **权重同步**：`rollout_probs_diff_mean` 与 eager 下相同，说明权重同步在 CUDA graph 下依然生效。graph 读取的是原地更新的权重缓冲区。
- **profile 文件**：每一步每个 rank 的 Proton 文件约 61 MB（eager 下约 106 MB），VizTracer 文件约 16–18 MB。
- **退出时的报错**：main_ppo 退出码为 0，但退出时有一条 DataLoader worker 被 kill 的 traceback。它出现在全部 5 步完成之后，发生在 Ray 关闭阶段。

## 产物

`s3://bench-artifacts-e00wpqevpr0037mwj6dvm8/rl-trace-observer/rl-trace-graph-20261002-132421-93518a4/`：`e0/`（包含 `profile_graph`、`profile_graph_2` 及两种合并结果）、`e2/step3.json.gz`、`e2/step4.json.gz`、`e2/artifacts/`。

## 尚未完成

- 在 Perfetto UI 中人工查看 graph 回放 kernel 与 `tokenspeed_generate` 的对应关系。
- 量化 CUPTI 常驻订阅的开销：session 停用期间，CUPTI 回调仍然注册着；不开 profile 的步骤未见明显变慢，但没有单独测量。
