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
