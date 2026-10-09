# 2026-10-08：改用 TokenSpeed fork 后的 GPU 回归

TokenSpeed 不再通过 `sitecustomize` 打补丁，改为从我们的 fork 安装：[Luosuu/tokenspeed](https://github.com/Luosuu/tokenspeed/tree/tianle/rl-trace) `tianle/rl-trace`，基于上游 `76fc28f`（即之前固定的 `0.1.0.post20260930` nightly）。fork 有两个提交：

- `eb558118`：`init_weights_update_group` 创建权重组时临时解除默认进程组的设备绑定，避免 torch 2.14 把它从 TokenSpeed 自己的世界 split 出来、trainer 连不上；
- `d81f1a25`：`TOKENSPEED_PROTON_SESSION_DIR`（scheduler 启动前建立常驻 Proton session，支持 CUDA graph）和 `TOKENSPEED_PROFILE_SAVE_IN_BACKGROUND`（`/stop_profile` 停止记录后立即返回，文件在后台线程写）。之后的 `0ea62714` 去掉了这个开关，后台写成为唯一行为。

本次运行时 server actor 为 `tokenspeed serve` 设置这两个变量。`tokenspeed-kernel` 仍是同一 commit 的 nightly `0.1.3.post20260930`。

复现方式：`TRAINER=one_step_off PLATFORM=gpu-h100-sxm scripts/gpu/submit_nebius.sh`（commit `54f1aac`，8×H100，job `rl-trace-fork-20261008-231050-54f1aac`）。

## 结果

- **fork 单元测试**：`test_request_handler_profile.py`、`test_proton_session.py`、`test_weights_update_group.py` 在 GPU 机器上 20 个全部通过（`tokenspeed_kernel` 没有 GPU 无法 import，所以只能在 GPU job 里跑）。
- **E0**（TokenSpeed 单独运行，TP=2）：32 项检查全部通过。eager 和 CUDA graph 两种模式都走常驻 Proton session，各 profile 两次：

  | 模式 | profile | `/stop_profile` 返回 (s) | 文件写完 (s) |
  |---|---|---|---|
  | eager | 1 | 0.42 | 3.42 |
  | eager | 2 | 0.25 | 3.25 |
  | graph | 1 | 0.14 | 3.15 |
  | graph | 2 | 0.16 | 3.16 |

  `/stop_profile` 在文件写完之前就返回；两个 TP rank 的 VizTracer 和 Proton 文件都完整写出，`rl-trace-merge` 与 `tokenspeed merge-traces` 都能合并。三种权重同步方式（TP=2 awake、VERL 顺序的 TP=2 与 TP=1）都通过置零/恢复检查，说明 fork 侧的权重组修复生效。
- **E2**（one-step-off，actor 4 卡，2 个 replica × TP=2）：5 步全部完成，`--strict --step 3` 与 `--step 4` 合并没有任何问题，每步 19 个产物。

  | step | step 时长 (s) | 等待数据 (s) | actor 训练 (s) | 权重同步 (s) | actor 的 `stop_profile` (s) |
  |---|---|---|---|---|---|
  | 1 | 20.7 | 9.9 | 10.5 | 0.16 | |
  | 2 | 3.6 | 0 | 3.5 | 0.17 | |
  | 3（profile） | 4.6 | 0 | 4.3 | 0.21 | 6.2 |
  | 4（profile） | 6.8 | 0 | 6.4 | 0.33 | 6.0 |
  | 5 | 7.8 | 0 | 7.4 | 0.42 | |

  与补丁版本（[2026-10-02](2026-10-02-tokenspeed-one-step-off.md)）一致：profile 步之后的下一步不等数据。actor 侧仍有约 6 s 的 Torch profiler 停止时间，与 TokenSpeed 无关。

## 复测：后台写成为唯一行为（fork `0ea62714`）

去掉 `TOKENSPEED_PROFILE_SAVE_IN_BACKGROUND` 开关后，在 8×H100 上重跑（commit `69d4a18`，job `rl-trace-fork-20261009-000315-69d4a18`）：

- fork 单元测试 20 个全部通过；
- E0 的 32 项检查全部通过，`/stop_profile` 在 0.14–0.32 s 内返回，约 3 s 后文件写完；
- E2 的 5 步全部完成，第 2–5 步等待数据均为 0 s，第 3、4 步的 strict 合并没有问题，每步 19 个产物。actor 的 `stop_profile` 仍约 6 s。

## 复测：review 修复之后（fork `58cdfadc`）

修复 PR #12 review 中确认的问题（失败路径上的长时间等待、weight sync 错误被掩盖、scheduled Torch profiler 的导出竞争、server 命名冲突等），fork 改为把 profile 文件写完后再改名到位，写失败时留下 `.failed` 标记。在 8×H100 上重跑（commit `da209ab`，job `rl-trace-review-20261009-085444-da209ab`）：

- fork 单元测试 22 个全部通过；
- E0 的 32 项检查全部通过；
- E2 的 5 步全部完成，第 2–5 步等待数据均为 0 s，第 3、4 步的 strict 合并没有问题，每步 19 个产物。
