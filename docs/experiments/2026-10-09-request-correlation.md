# 2026-10-09：request-level correlation（P6）的 GPU 验证

计划与进展见 Luosuu/rl-trace-observer#14，代码在 Luosuu/rl-trace-observer#15。

## P6-0：TokenSpeed 是否保留请求的 `rid`

8×H100，job `rl-trace-p6-0-20261009-131621-64ca25c`，TokenSpeed fork `58cdfadc`。eager 与 CUDA graph 两种模式结果一致：

- 响应的 `meta_info.id`、gRPC servicer、AsyncLLM（`rid` 与 `user_rid`）和 scheduler（`Req: <rid> Finish!`）用的都是我们发出的 `rid`；
- 请求不带 `rid` 时，gateway 会自己生成一个（`gnt-...`），server actor 总会带上 `rid`。

因此 request ID 采用 VERL 每次调用发给 server 的 uuid。E0 把这一点作为检查（`request_id_passes_through`）。

另外发现：按 `rid` abort 一个运行中的请求时，scheduler 确实结束了它，但这个请求的 HTTP 调用一直没有返回（约 600 s 后 gateway 返回 500）。这与关联无关，记为后续事项。

## P6-c：从请求跟踪到 GPU kernel

8×H100，job `rl-trace-p6c-20261009-182739-bd09e03`（commit `bd09e03`，TokenSpeed fork `2a23f039`），hybrid E2：GRPO，Qwen2.5-0.5B，2 个 replica × TP=2，CUDA graph，第 3、4 步 profile。

链路：

1. agent loop 每次调用记录 `rollout_request`；
2. server actor 记录 `tokenspeed_generate`，带同一个 `request_id`；
3. fork 在 profile 期间为每次 forward 记录 `forward_batch`（`tokenspeed::forward` 线程，参数 `request_ids`）；
4. 在 `forward_batch` 内，eager kernel 的 scope 和 CUDA graph replay 的 scope（fork 新增的 `cuda_graph.replay`）都有 VizTracer→Proton flow，Proton 再把这些 scope 连到 GPU kernel。

`scripts/gpu/check_request_flows.py` 对每个合并后的 step 统计如下（两步结果相同）：

| | step 3 | step 4 |
|---|---|---|
| 该步 server 处理的请求 | 128 | 128 |
| 从 agent loop 连过来 | 128 | 128 |
| 连到列有该请求的 forward（首次与最后一次） | 128 | 128 |
| 首次 forward（prefill，eager）能连到 GPU kernel | 128 | 128 |
| 最后一次 forward（decode，CUDA graph 回放）能连到 GPU kernel | 128 | 128 |
| flow 经过未服务该请求的 forward | 0 | 0 |

此前的 trace 中，CUDA graph 下 TP0 的 516 次 forward 只有 2 次（eager prefill）有到 Proton 的 scope flow：回放 graph 不执行 Python，没有 scope，decode 请求因此连不到 kernel。fork 给 graph 回放加上 scope 后，decode 请求也能连到 kernel。

其余检查：E0 的 34 项检查（含 `request_id_passes_through`）全部通过；第 3、4 步的 `--strict` 合并没有问题，每步选中 27 个产物；fork 的单元测试在 GPU 上全部通过。

## P6-d：请求的权重版本与消费它的训练 step

8×H100，job `rl-trace-p6d-20261009-230600-06e7462`（commit `06e7462`），one-step-off E2：actor 用 4 张卡，2 个 standalone replica × TP=2，第 3、4 步 profile。

| | step 3 | step 4 |
|---|---|---|
| 请求数 | 128 | 128 |
| 带权重版本的请求 | 128 | 128 |
| 两端权重版本不一致 | 0 | 0 |
| 请求按 (batch_step, weight_version) | 3/3：128 | 4/4：128 |
| 该步窗口内的 `rollout_batch` | 第 3 步请求，第 4 步训练 | 第 4 步请求，第 5 步训练 |

第 n 步请求的 batch 用第 n 步同步的权重（版本 n）生成，由第 n+1 步训练，staleness 为 1。请求的跟踪链路（agent loop → server → forward → kernel）的检查也全部通过。

CPU 端到端测试覆盖了完整序列：第 1 步之前请求的 batch 用版本 0 的权重、在第 1 步训练；其后第 n 步请求的 batch 在第 n+1 步训练。同步 trainer 中，一个 step 训练自己生成的 batch，权重版本为 step − 1。

### 权重版本 counter track

8×H100，job `rl-trace-p6d-counter-20261011-101626-bce1ae8`（commit `bce1ae8`），配置同上。每次 `CheckpointEngineManager.update_weights` 记录为 trainer 上 `trainer/weights` lane 的 `weight_update` span，合并后成为 trainer 进程上的 `rollout weight_version` counter track。

| | step 3 | step 4 |
|---|---|---|
| counter 取值（按时间） | 2 → 3 | 3 → 4 |
| 请求的权重版本 | 3 | 4 |

窗口开始时 server 持有上一步同步的权重（n−1）；第 n 步中途同步到版本 n，随后生成的下一 batch 的请求都是版本 n。其余检查（链路、两端版本一致）全部通过。同步 trainer 在 `step()` 返回后才同步权重，因此单步 trace 中 counter 全程为 n−1（CPU 端到端测试覆盖）。

### Perfetto 导入无告警

此前合并的 step trace 在 Perfetto 中有约 5.4 万条 `slice_spill_overlapping_complete_event`：同一 GPU stream 上约 17% 的 kernel 与前一个重叠 1–3 µs（programmatic dependent launch：FlashAttention 的 `prepare_varlen_num_blocks` → 主 kernel、flashinfer 的 lamport all-reduce → RMSNorm、采样 kernel、cuBLAS split-K），另有 3 条是 fork 的 `forward_batch` 与 VizTracer 记录的 `contextmanager.__exit__` 部分重叠。merger 现在把与本 lane 上 slice 部分重叠的 kernel 移到该 stream 的 overlap lane（完全嵌套的保留，例如 Kineto 的 `gpu_user_annotation`），launch flow 随之移动；fork 在 `device._forward` 中用 tracer 的 C `getts` 取时间戳。

job `rl-trace-overlaps-20261016-150754-07cb998`（fork `3314732f`）：第 3、4 步 trace_processor 无 error/data_loss 统计，各约 5.4/5.8 万个 kernel 在 overlap lane 上；请求链路与权重版本检查全部通过。

## Profile 步末尾的空白：Torch profiler 停止

profile 步在 actor 最后一个 kernel 之后约 5.8 s 没有任何记录，VERL 的 `timing_s/stop_profile` 与之相符：driver 在等各训练 worker 停止 Torch profiler。现在每次停止是 worker 上 `rank_<r>/profiler` lane 的 `torch_profiler_stop` span（带各部分耗时），driver 的等待是 `trainer/profiler` lane 的 `stop_profile`，agent loop worker 每批生成是 `agent_loop/batch` lane 的 `agent_loop_batch`（生成在 2.5–3.1 s 完成，one-step-off trainer 要等 actor 训练返回才取走，因此 `rollout_batch` 结束得晚）。`scripts/gpu/summarize_step_gaps.py` 逐步汇总这些时间。

one-step-off E2，第 3、4 步（ms）：

| 配置 | commit | driver `stop_profile` | 其中 Kineto disable | 交给导出线程 | 每个 worker 的 Torch trace |
|---|---|---|---|---|---|
| Kineto，记录全部 aten op（此前） | `36dc483` | 5846 / 5969 | 3.9 s | 某个 worker 1.9–2.0 s | 20 MB |
| `cupti_monitor`，全部 aten op | `36dc483` | 3102 / 3106 | 2.3 s | 0.8 s（同步写 CPU 事件） | 26 MB |
| Kineto + torch 的 Python writer | `6918797` | 4746 / 4651 | 4.0 s | 0.6–0.7 s | 17 MB |
| Kineto，只记录 marker | `b4c3b7b` | 1649 / 1579 | 1.5 s | < 1 ms（一次 111 ms） | 3.1 MB |
| `cupti_monitor`，只记录 marker | `b4c3b7b` | **14.5 / 13.1** | 8 ms | 3 ms | 8.2 MB |

- 交给导出线程的约 2 s 来自 Kineto 的 `save` 全程持有 GIL：导出线程先拿到 GIL 时，训练线程卡在 `thread.start()` 直到写完（本地复现：`thread.start()` 421 ms，与 save 时长相同）。GC 为 0，复制 profiler 约 0.05 ms。改用 torch 的 Python writer 后仍有 0.6–0.7 s（列出 activity 时持有 GIL），但不再是整个写入。
- 每个 worker 每步约 68 万个 CPU op 事件（其中 `aten::as_strided` 15 万），远多于 10.5 万个 kernel；没有 call stack 和 input shape。只记录 RecordFunction 的 user scope 后，VERL 的 marker、CUDA launch 与 kernel 都保留，Kineto disable 从 3.9 s 降到 1.5 s，`cupti_monitor` 下几乎为 0。`RL_TRACE_TORCH_CPU_OPS=1` 恢复记录 aten op。
- 所有配置下第 3、4 步 strict 合并无问题，请求链路检查全部通过（128/128 请求连到 kernel），训练 kernel 都在 trace 中。
