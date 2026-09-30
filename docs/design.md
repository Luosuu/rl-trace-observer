# Global RL Trace Bridge 设计文档

## 1. 项目状态

本项目希望为分布式强化学习训练提供统一的 Chrome Trace / Perfetto 视图，将训练、推理和 GPU kernel 的异构 profiler 输出放到同一条全局时间线上。

当前代码是组件级 PoC，已经验证：

- RL-Insight trace event 可以转换成增量 Chrome Trace JSONL；
- 可选 monkey patch 可以围绕 VERL `DistProfiler` profiling window 启停 actor 侧 VizTracer；
- observer 异常不会中断原 profiler backend；
- monkey patch 支持幂等安装和安全卸载。

当前尚未形成端到端 Global RL Trace Bridge。特别是 worker backend 选择、artifact manifest、全局身份映射、时钟校准、TokenSpeed 控制和最终 merger 仍待实现。

## 2. 背景与问题

一次 VERL RL 训练通常跨越多类进程和 profiler：

| 领域 | 典型进程 | 主要 profiler | 能回答的问题 |
| --- | --- | --- | --- |
| Actor / Critic 训练 | PyTorch distributed workers | Torch Profiler | operator、CUDA launch、训练 kernel |
| RL 阶段语义 | trainer、actor、rollout workers | RL-Insight | actor update、log-prob、generate 等阶段 |
| Rollout Python 调度 | TokenSpeed scheduler ranks | VizTracer | Python 调度、请求处理、scope |
| Rollout GPU kernel | TokenSpeed scheduler ranks | Proton | Triton/kernel timeline |

这些 profiler 分别输出不同文件，使用不同 PID/rank 命名和时间基准。仅仅把 JSON 拼接起来会产生以下问题：

- worker 上的语义 span 可能没有进入自定义 collector；
- 不同节点可能拥有相同 OS PID；
- `rank_0`、`replica_0` 等 lane 名在不同角色之间重复；
- wall clock、Torch、VizTracer 和 Proton 的时间原点不同；
- artifact 分散在多个节点，缺少完整性和归属信息；
- 时间接近不能证明 actor 请求与 rollout/kernel 之间存在因果关系。

## 3. 项目目标

### 3.1 Session-level MVP

首个可交付目标是在一次 profiling session 内生成单个可由 Perfetto 打开的 trace，至少包含：

- 一个或多个 actor Torch Profiler trace；
- VERL/RL-Insight 的 semantic lanes；
- 多个 TokenSpeed rank 的 VizTracer Python trace；
- 多个 TokenSpeed rank 的 Proton GPU kernel trace；
- 稳定、无冲突的 process/thread identity；
- 可验证的时间归一化和 clock-skew 诊断；
- manifest 中声明的 artifact 完整性状态。

### 3.2 Request-level Bridge

第二阶段在 session-level 视图上增加因果关系：

- 为 rollout 请求生成全局 `trace_id/request_id`；
- 从 actor/driver 传播到 rollout replica；
- 在 RPC 两侧写入匹配的 Chrome flow event；
- 将 rollout Python scope 继续连接到 Proton kernel scope。

### 3.3 最小侵入原则

- VERL core 默认零修改；
- 优先复用 VERL 已有的 RL-Insight 埋点；
- TokenSpeed 和 VERL 不互相依赖；
- 本项目作为独立 integration package 依赖两者；
- actor VizTracer 仅作为可选功能启用薄 monkey patch；
- 只有外部方案无法可靠实现时，才考虑向上游增加通用 hook。

## 4. 非目标

首个 MVP 不计划：

- 替代 Torch Profiler、RL-Insight、VizTracer 或 Proton；
- 在 Perfetto 中实现在线流式展示；
- 在无任何同步信息时自动修复任意跨节点时钟漂移；
- 推断没有显式 ID 的 request-level 因果关系；
- 修改 VERL `ProfilerConfig.tool` 为多 profiler 列表。

## 5. 总体架构

```mermaid
flowchart LR
    V["VERL existing annotations"] --> RI["RL-Insight trace_state"]
    RI --> TC["RL Trace JSONL client"]
    TC --> RJ["Semantic span JSONL"]
    TC -. optional .-> RH["RL-Insight Ray backend"]

    AT["Actor Torch Profiler"] --> AC["Artifact collector"]
    RJ --> AC

    VR["VERL external RolloutReplica"] --> TS["TokenSpeed control API"]
    TS --> VZ["VizTracer per rank"]
    TS --> PR["Proton per rank"]
    VZ --> AC
    PR --> AC

    AC --> MF["Session manifest"]
    MF --> GM["Global trace merger"]
    GM --> PF["One Perfetto / Chrome trace"]
```

## 6. 关键实现设计

### 6.1 RL-Insight 语义事件采集

VERL 已经在 actor 和 rollout 路径调用 `RLInsightLogger.trace_state`。本项目不重复 patch 这些业务函数，而是注册 RL-Insight monitor client。

当前实现注册了名为 `rl_trace_observer` 的 backend，并将 trace event 写成 append-only JSONL。增量写入避免依赖 Ray worker 的优雅退出。

但 VERL worker 首次调用 `trace_state` 时可能执行无参数 `rl_insight.init()`，从而选择默认 `ray` backend，而不是 driver trainer config 中的自定义 backend。

MVP 方案：

1. 在 RL-Insight fork（`Luosuu/rl-insight@tianle/server-backend-env`，已向上游提交）中增加 `RL_INSIGHT_SERVER_BACKEND`，覆盖 `server.backend`；
2. 插件在每个进程加载时注册 `rl_trace_observer` backend，并默认设置 `RL_INSIGHT_SERVER_BACKEND=rl_trace_observer`（用户显式设置时不覆盖）；
3. worker 即使无参数初始化，也会进入本地 collector；
4. 若用户启用原 RL-Insight 服务，event 同时转发给 RL-Insight 内置 Ray client；
5. RL-Insight 不支持该变量时 fail fast，不提供替换默认 `ray` factory 的回退；
6. 通过真实 Ray actor 与 VERL `RayWorkerGroup` 测试验证 backend 选择。

项目由 uv 管理并固定 Python 3.12；verl、RL-Insight fork（按 commit 固定）与 VizTracer 为根依赖，TokenSpeed 为唯一 extra。Ray 运行遵循 Ray + uv 范式：`ray_runtime_env.yaml` 上传项目为 `working_dir`，并以 `uv run --locked --extra tokenspeed python` 作为 `py_executable`，每个 worker 使用同一锁定环境。末尾的 `python` 不可省略：否则 `uv run <default_worker.py>` 会从 Ray 安装位置而不是 `working_dir` 发现项目。

### 6.2 Actor profiler

Actor 继续使用 VERL 原生 Torch Profiler。collector 根据 session manifest 发现其 trace 文件。

默认不在 actor 启用 VizTracer，因为：

- Torch trace 已经提供 operator/CUDA 信息；
- RL-Insight 提供阶段语义；
- 少一个 profiler 可以降低训练扰动。

若需要 Python call stack，可设置 `RL_TRACE_VIZTRACER=1`，通过可逆 monkey patch 包围 `DistProfiler.start/stop`。正式使用前需要补充重复 start、nested lifecycle、多 profiler instance 和异常退出状态机。

### 6.3 TokenSpeed rollout profiler

独立项目通过 VERL external module 注册 TokenSpeed `RolloutReplica` 和 `ServerAdapter`，不修改 VERL trainer。

在现有 `start_profile(profile_step=...)` 调用中生成确定性的 session context，并请求 TokenSpeed：

```json
{
  "activities": ["VIZTRACER", "PROTON"],
  "output_dir": "<session>/rollout/<replica>/<rank>"
}
```

TokenSpeed 内部负责每个 rank 的 Python scope → Proton scope 连接。多 rank 的时间对齐、PID/flow-ID 隔离仍由 merger 完成。

### 6.4 TraceContext

跨进程 context 必须可序列化且可确定性重建：

```text
run_id
profile_session_id
global_step
role
hostname
os_pid
framework_rank
replica_rank
request_id        # request-level 阶段
clock_snapshot_id
```

首版 `profile_session_id` 可以由 `run_id + global_step` 生成。不能只使用 OS PID、rank 或文件名推断 session。

**当前实现（`rl_trace_observer.context`，schema_version 1）**：

- `TraceContext` 是可 JSON 序列化的 frozen dataclass，包含上述字段（`clock_snapshot_id` 暂由进程记录的 clock snapshot 代替）；`from_json` 拒绝其他 schema 版本。
- `run_id`：优先取 `RL_TRACE_RUN_ID`；否则取 Ray `get_session_name()` 与 `get_job_id()`，即 `<session_name>-job-<job_id>`。同一训练 job 的所有 Ray 进程天然一致，无需额外传播；只有 job id 时不同集群会重复（都从 `01000000` 开始），所以带上 session 名。非 Ray 进程（如 P2 的 TokenSpeed server）需显式传入。
- `global_step`：Torch/VizTracer 从文件名得到（VERL `profile_step`）；RL-Insight span 不带 step，因此 driver 在 VERL v1 trainer 的 `PPOTrainer.step` 外包一层（模块导入时通过 post-import hook 打补丁，不在插件加载时导入重量级模块），每个 step 在 `trainer` lane 上记录一个 `global_step` span（args 含 `global_step`、`run_id`），作为该 profile session 的时间窗口。异步 trainer 中下一步的 generation 可能与当前 step 重叠，精确归属留给 P6 request-level。
- `role`：进程级 role 目前只有 trainer（记录 `global_step` 的进程）；artifact 级 role 来自 Torch 文件名前缀和 VizTracer 的 role。

### 6.5 Artifact manifest

每个进程写入 artifact record，driver/collector 汇总为 session manifest：

```json
{
  "schema_version": 1,
  "run_id": "ppo-run-001",
  "profile_session_id": "ppo-run-001-step-42",
  "global_step": 42,
  "artifacts": [
    {
      "kind": "rl_insight_jsonl",
      "hostname": "node-1",
      "os_pid": 1234,
      "role": "actor",
      "rank": 0,
      "path": ".../rl-insight-node-1-pid-1234.chrome.jsonl",
      "complete": true
    }
  ]
}
```

manifest 还需要记录 profiler 版本、文件大小、checksum、写入完成状态、clock snapshot 和缺失 artifact。

节点本地目录不能假设对 driver 可见。collector 必须支持共享文件系统路径或显式上传/回传。

**当前实现（schema_version 1）**：

- 每个写 artifact 的进程在 `RL_TRACE_OUTPUT_DIR` 写 `rl-trace-process-<host>-pid-<pid>.json`（原子替换），记录 hostname、pid、Ray job/node/worker/actor id 与 actor 名、`torch.distributed` rank/world_size、首次写入时的 wall/monotonic clock snapshot，以及它登记的 artifact。RL-Insight client 创建时与 VizTracer 启动时写入；此时 VERL worker 的 process group 已初始化。RL-Insight JSONL 在 client 创建时即创建，空文件表示该进程没有 span。
- `rl-trace-merge` 在合并前构建 session manifest，并输出 `<output>.manifest.json`：进程列表、每个 artifact 的 size/sha256/所属进程/rank/完整性，以及 `incomplete`、`duplicate`、`missing`、`unlinked`、`ambiguous` 问题（`duplicate` 指文件名与内容都相同的副本）。`--strict` 下任一问题都会失败；合并失败时仍写出 manifest，并删除输出路径上旧的 trace。
- Torch/VizTracer 文件名只含 pid，按 pid 关联进程记录；多个 host 复用同一 pid 时用 rank（Kineto `distributedInfo` 或文件名）区分。
- 进程记录还包含 `run_id`、`role` 和已加载框架/profiler 的版本（`versions`）；artifact 条目可带 `global_step`。manifest 增加 `runs` 与 `sessions`（每个 profile session 的时间窗口与 step 专属 artifact），以及 `mixed_runs`（多个 run 且未用 `--run` 选择）、`no_step_window`（所选 step 没有 `global_step` span）问题。
- session 暂定为输入目录下的全部 artifact（每次运行使用独立的 `RL_TRACE_OUTPUT_DIR`）；`run_id` / `profile_session_id` / `global_step` 与 profiler 版本尚未记录。显式 artifact 回传尚未实现，目前依赖共享文件系统。

### 6.6 全局 identity

Chrome trace 中不得直接复用跨节点 OS PID。merger 建立稳定映射：

```text
(hostname, os_pid, role, framework_rank, replica_rank)
    -> synthetic numeric pid/tid
```

并输出：

- `process_name` metadata；
- `thread_name` metadata；
- 原 hostname/PID/rank/role 作为 args；
- 独立的 PID namespace，避免与 TokenSpeed 固定 PID 和 flow-ID 冲突。

### 6.7 时间模型

RL-Insight 的 `start_time_ns/end_time_ns` 来自各进程的 `time.time_ns()`。这提供 Unix epoch wall-clock timestamp，但不提供跨节点时钟同步。

每个 artifact 应记录：

```text
wall_time_ns
monotonic_time_ns
hostname
clock_source
clock_snapshot_id
```

merger 负责：

1. 读取 Torch/VizTracer/Proton 的 absolute anchor；
2. 选择统一 `global_base_time_ns`；
3. 将所有 timestamp 转换为相对 global base 的微秒；
4. 检测负 duration、anchor 不一致和明显 clock skew；
5. 在无法可信对齐时告警或拒绝生成误导性 flow。

生产集群应使用 NTP/PTP。后续可以增加 driver↔worker handshake 估算 offset，但不能把网络延迟估算等价为严格时钟同步。

### 6.8 Merger

Global merger 的输入是 manifest，而不是无约束 glob：

```text
manifest
  ├── actor Torch traces
  ├── RL-Insight semantic JSONL
  ├── TokenSpeed VizTracer traces
  └── TokenSpeed Proton traces
```

处理顺序：

1. 校验 manifest schema 和 artifact 完整性；
2. 解析各 profiler 的 absolute anchor；
3. 计算 global base 和 clock diagnostics；
4. 分配 synthetic PID/TID；
5. 复用 TokenSpeed rank-local VizTracer→Proton flow 逻辑；
6. 合并 actor Torch 和 RL-Insight semantic lanes；
7. 重写 flow IDs，保证跨 rank/role 唯一；
8. 写出一个标准 Chrome Trace JSON；
9. 用 Perfetto trace processor 或浏览器加载测试验证格式。

## 7. 可靠性与性能

- observer/collector 失败默认不能中断训练，但必须产生可观察 warning；
- JSONL 单条 append 使用进程内锁；
- 多进程不共享同一文件；
- trace output 必须包含 hostname 和 session，避免覆盖；
- artifact 写入完成需要原子 marker 或 manifest 状态；
- merger 对缺失文件提供 strict/best-effort 两种模式；
- profiling overhead 必须分别测量 Torch-only、RL-Insight、VizTracer+Proton 和组合配置。

## 8. 兼容性策略

首个 MVP 固定并记录已验证版本：

- VERL commit/version；
- RL-Insight version；
- TokenSpeed commit/version；
- VizTracer、Torch、Proton version。

启动时检查：

- RL-Insight registry 和 client factory signature；
- VERL external module 和 profiler method signature；
- TokenSpeed profiling control API；
- TokenSpeed trace anchor/schema。

不满足契约时 fail fast，并输出明确的兼容性错误。

## 9. Work plan

### P0：修复真实 worker 采集链路

- [x] 通过 `RL_INSIGHT_SERVER_BACKEND`（RL-Insight fork）为无参数初始化的 worker 选择本地 backend；
- [x] 可选转发到 RL-Insight 内置 Ray client，转发失败不影响训练；
- [x] 验证 driver、actor、rollout 进程均加载插件（CPU 上使用 VERL 真实 `RayWorkerGroup`；rollout 以调用同一 `RLInsightLogger.trace_state` 的 Ray actor 代替 vLLM/SGLang server）；
- [x] 增加真实 RL-Insight lazy-init 测试；
- [x] 增加单机多 Ray actor smoke test。
- [x] CPU 上运行真实 `verl.trainer.main_ppo` 一个 PPO step（FSDP2 actor + 测试用 mock rollout，`tests/test_cpu_ppo.py`），actor 与 rollout 进程均写出 semantic artifact，合并后 RL-Insight 与 Torch 的 `actor_update` 在 Perfetto 中对齐；
- [ ] 在真实 vLLM/SGLang/TokenSpeed rollout server 中确认 `*_generate` span（需 GPU）。

已验证版本：`verl==0.9.1`、`tokenspeed==0.1.0`（`transformers` 覆盖为 5.12.0）、RL-Insight fork `72763be`（基于 0.3.0）。VERL 0.9 通过 `verl.plugins` entry point 在每个导入 verl 的进程中自动加载插件，并通过 `get_ppo_ray_runtime_env` 将 `VERL_RL_INSIGHT_ENABLE` 转发给所有 worker。两个版本的 `load_monitor_config` 都只允许环境变量覆盖 `server.url`，无法覆盖 `server.backend`，因此项目固定使用增加了 `RL_INSIGHT_SERVER_BACKEND` 的 fork。

完成标准：至少两个 Ray worker 的 `trace_state` 都产生本地 semantic artifact。

### P1：TraceContext 与 artifact manifest

- [x] 定义 versioned TraceContext schema（`run_id` 由 Ray session + job 确定性重建，可用 `RL_TRACE_RUN_ID` 覆盖；P6 request-level 前置）；
- [x] 定义 versioned manifest schema（进程记录与 session manifest，schema_version 1）；
- [x] 为 artifact 增加 hostname、rank、Ray actor、checksum；
- [x] 为 artifact 增加 role、run_id/profile_session_id、profiler 版本；manifest 按 run 与 profile session 分组，`rl-trace-merge --run/--step` 选择；
- [x] 支持共享目录；
- [ ] 支持显式 artifact 回传（节点本地目录）；
- [x] 检测缺失、重复和未完成 artifact，并关联 Torch/VizTracer trace 到所属进程。

完成标准：driver 可以列举一次 session 的全部预期和实际 artifact。

### P2：TokenSpeed external rollout integration

- [ ] 注册 TokenSpeed `RolloutReplica`；
- [ ] 注册 TokenSpeed `ServerAdapter`；
- [ ] 转换 VERL profiling context；
- [ ] 调用 `/start_profile` 和 `/stop_profile`；
- [ ] 启用 `VIZTRACER + PROTON`；
- [ ] 将各 rank artifact 写入 manifest。

完成标准：不修改 VERL core，两个 TokenSpeed rank 能产生匹配的 VizTracer/Proton artifact。

### P3：Session-level global merger

- [x] 实现 RL-Insight JSONL reader；
- [x] 实现 actor Torch trace reader（VERL `build_trace_basename` 命名，`baseTimeNanoseconds` 锚点）；
- [x] 实现 actor VizTracer reader（`viztracer_metadata.baseTimeNanoseconds` 锚点）；
- [ ] 接入 TokenSpeed multi-rank merge（`tokenspeed.cli.trace_merge.merge_all_ranks`）；
- [x] 实现 global base time（整数纳秒锚点，避免 epoch 纳秒超出 float64 精度）；
- [x] 实现 synthetic PID/TID allocator（TID 全局唯一：Perfetto JSON importer 仅按 tid 识别线程）；
- [x] 实现全局 flow-ID namespace（按 source 重新编号）；
- [x] 输出 process/thread metadata；
- [x] 生成单个 Chrome Trace JSON（`rl-trace-merge`），并用 Perfetto trace_processor 验证；
- [ ] 同一进程的 RL-Insight lane 与 Torch trace 归并到同一 process（Torch trace 不含 hostname，需 manifest）。

完成标准：Perfetto 中同时显示 actor Torch、semantic lanes、rollout Python 和 Proton kernel。当前已覆盖 actor Torch、semantic lanes 与 actor VizTracer；CPU 测试中三者对同一操作的时间嵌套一致，误差在亚毫秒级。

### P4：时钟诊断

- [ ] 记录 wall/monotonic clock snapshot；
- [ ] 检测负 duration 和 wall-clock jump；
- [ ] 检测跨节点明显 skew；
- [ ] 在 trace 和报告中显示时钟质量；
- [ ] 评估 driver↔worker offset handshake。

完成标准：错误时钟不会被静默当成可信全局顺序。

### P5：可选 actor VizTracer 稳定化

- [ ] 增加进程级 ownership lock；
- [ ] 实现 `IDLE → STARTED → STOPPED` 状态机；
- [ ] 处理重复/nested start；
- [ ] 测试多个 `DistProfiler` 实例；
- [ ] 测试取消和异常退出。

完成标准：启用 actor VizTracer 不会泄漏或覆盖 lifecycle。

### P6：Request-level correlation

- [ ] 生成并传播 request trace ID；
- [ ] 在 actor/RPC 发送侧写 flow start；
- [ ] 在 rollout 接收侧写 flow finish；
- [ ] 将 request ID 加入 RL-Insight labels；
- [ ] 验证 flow 可继续连接 TokenSpeed Proton kernel。

完成标准：可以从一个 actor rollout 请求跟踪到目标 replica 和相关 kernel。

### P7：兼容性、CI 与发布

- [ ] 固定首个兼容版本矩阵；
- [ ] 增加 API contract tests；
- [ ] 增加多进程 end-to-end fixture；
- [ ] 验证 Perfetto 可解析性；
- [ ] 测量 profiling overhead；
- [ ] 补充安装、故障排查和示例配置。

## 10. MVP 验收场景

最小 vertical slice：

```text
1 个 VERL driver
2 个 Ray worker
1 个 actor Torch trace
2 个 TokenSpeed rollout ranks
2 对 VizTracer/Proton artifacts
N 个 RL-Insight semantic JSONL
1 个 session manifest
1 个最终 Perfetto trace
```

验收检查：

- 每个 worker 的 semantic lane 可见；
- actor Torch operator/CUDA lane 可见；
- 两个 rollout rank 的 Python lane 可见；
- Proton kernel lane 可见；
- PID/TID 无跨节点冲突；
- trace 时间顺序通过 anchor/skew 检查；
- 缺失 artifact 会产生明确错误；
- VERL core 无修改。

## 11. 开放问题

- RL-Insight 上游是否接受 `RL_INSIGHT_SERVER_BACKEND`？合入发版后改回 PyPI 依赖。
- TokenSpeed artifact 最终由 rollout replica 返回，还是由独立 collector 扫描共享目录？
- 首版是否要求多节点 PTP，还是允许 NTP + skew warning？
- actor VizTracer 是否值得默认支持，还是保持 Torch + RL-Insight 即可？
- request ID 应复用推理框架 ID，还是由 RL Trace Bridge 生成全局 ID？
