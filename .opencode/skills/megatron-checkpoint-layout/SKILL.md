---
name: megatron-checkpoint-layout
description: Bilingual guidance for Megatron checkpoint 1D 2D 3D mp_rank layouts across tensor pipeline and expert parallel dimensions
compatibility: opencode
metadata:
  domain: distributed-training
  framework: megatron
  repo: llava-onevision2
---

## Purpose / 用途

Use this skill when diagnosing, designing, or converting Megatron/Megatron-Core checkpoints that may use TP, PP, and EP.

在排查、设计或转换使用 TP、PP、EP 的 Megatron / Megatron-Core checkpoint 时，使用这个 skill。

## Core rule / 核心规则

- TP only: `mp_rank_{tp}`
- TP + PP: `mp_rank_{tp}_{pp}`
- TP + PP + EP: `mp_rank_{tp}_{pp}_{ep}`

- 只有 TP：`mp_rank_{tp}`
- TP + PP：`mp_rank_{tp}_{pp}`
- TP + PP + EP：`mp_rank_{tp}_{pp}_{ep}`

The key discriminator is whether expert parallelism participates in checkpoint sharding.

真正的分界点是：expert parallelism 是否参与了 checkpoint 切分。

- If EP is present, treat the checkpoint layout as 3D.
- If EP is absent, treat the checkpoint layout as non-EP and use 1D or 2D.

- 如果存在 EP，就按 3D 布局处理。
- 如果不存在 EP，就按非 EP 布局处理，即 1D 或 2D。

## Mental model / 心智模型

Megatron does not treat `pp > 1` as meaning 3D by itself.

Megatron 不会因为 `pp > 1` 就自动把 checkpoint 视为 3D。

- PP adds a pipeline index.
- EP adds an expert index.
- The third coordinate exists because EP exists, not because PP exists.

- PP 只是在目录里增加 pipeline 这一维。
- EP 才会增加 expert 这一维。
- 第三维存在的原因是 EP 存在，而不是因为 PP 存在。

So even if `tp=1` and `pp=1`, once EP is enabled the checkpoint naming is still conceptually 3D because ranks are addressed by `(tp, pp, ep)`.

所以即使 `tp=1` 且 `pp=1`，只要启用了 EP，checkpoint 在语义上仍然是 3D，因为 rank 仍然由 `(tp, pp, ep)` 共同定位。

## Practical interpretation / 实际使用解释

When reading or converting checkpoints:

在读取或转换 checkpoint 时：

1. First decide whether EP exists in the checkpoint contract.
2. If EP exists, require `mp_rank_{tp}_{pp}_{ep}`.
3. If EP does not exist, read as `mp_rank_{tp}` or `mp_rank_{tp}_{pp}`.
4. Do not infer 3D solely from `pipeline_model_parallel_size > 1`.

1. 先判断这个 checkpoint 契约里是否存在 EP。
2. 如果存在 EP，就要求目录是 `mp_rank_{tp}_{pp}_{ep}`。
3. 如果不存在 EP，就按 `mp_rank_{tp}` 或 `mp_rank_{tp}_{pp}` 去读。
4. 不要仅凭 `pipeline_model_parallel_size > 1` 就推断它一定是 3D。

## Typical failure pattern / 典型错误模式

Bad assumption:

错误假设：

- `pp > 1` so loader chooses a 3D reader.

- 只要 `pp > 1`，loader 就应该走 3D reader。

Why it fails:

为什么会失败：

- Dense non-MoE checkpoints with TP+PP usually use `mp_rank_{tp}_{pp}` only.
- A 3D loader then looks for an EP coordinate that is not present.

- 普通 dense、非 MoE 的 TP+PP checkpoint，通常只有 `mp_rank_{tp}_{pp}`。
- 这时 3D loader 会去找并不存在的 EP 坐标，最终报错。

## Recommended repo-local contract / 当前仓库建议契约

For this repository, follow this rule:

这个仓库建议遵循以下规则：

- if `expert_parallel_size` is passed, require 3D
- if `expert_parallel_size` is not passed, use non-EP loading

- 如果传了 `expert_parallel_size`，就强制按 3D 处理
- 如果没有传 `expert_parallel_size`，就走非 EP 的加载逻辑

This matches Megatron's path-building logic better than using `pp > 1` as the branch condition.

这个规则比“用 `pp > 1` 作为分支条件”更贴近 Megatron 自己的路径生成逻辑。

## What to check during debugging / 调试时要检查什么

- What are the actual shard directory names under the checkpoint root?
- Was `expert_parallel_size` provided by the caller?
- Is the model dense or MoE?
- Is the loader branching on EP or incorrectly branching on PP?
- If conversion failed, which exact `mp_rank_*` pattern was expected and which one exists on disk?

- checkpoint 根目录下，实际 shard 目录名是什么？
- 调用方是否传入了 `expert_parallel_size`？
- 当前模型是 dense 还是 MoE？
- loader 是按 EP 分支，还是错误地按 PP 分支？
- 如果转换失败，程序期望的 `mp_rank_*` 模式是什么，磁盘上实际又是什么？

## Expected outputs when using this skill / 使用本 skill 时的期望输出

When asked to analyze a checkpoint issue, return:

当你被要求分析 checkpoint 问题时，应该返回：

1. the inferred layout class: 1D, 2D, or 3D
2. the reason for that classification
3. the expected directory naming pattern
4. whether the caller should use a non-EP loader or an EP-aware loader
5. any mismatch between runtime arguments and on-disk shard layout

1. 推断出的布局类别：1D、2D 或 3D
2. 这样分类的原因
3. 期望的目录命名模式
4. 调用方应使用非 EP loader 还是 EP-aware loader
5. 运行时参数与磁盘上 shard 布局之间是否存在不匹配
