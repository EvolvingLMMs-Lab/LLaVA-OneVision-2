---
name: llava-onevision2-consistency
description: Bilingual guide for running and interpreting LLaVA-OneVision2 HF vs Megatron consistency checks across TP and PP settings
compatibility: opencode
metadata:
  domain: model-validation
  framework: llava-onevision2
  repo: llava-onevision2
---

## Purpose / 用途

Use this skill when validating whether a HuggingFace checkpoint and a Megatron/MCore checkpoint are behaviorally consistent in this repository.

在这个仓库里，需要验证 HuggingFace checkpoint 和 Megatron/MCore checkpoint 是否行为一致时，使用这个 skill。

This skill is specifically for:

- `examples/llava_onevision2/check_model_consistency.sh`
- `examples/llava_onevision2/check_model_consistency.py`
- TP/PP combinations such as `tp1pp1`, `tp2pp1`, `tp1pp2`, `tp2pp2`

这个 skill 专门用于：

- `examples/llava_onevision2/check_model_consistency.sh`
- `examples/llava_onevision2/check_model_consistency.py`
- `tp1pp1`、`tp2pp1`、`tp1pp2`、`tp2pp2` 这类 TP/PP 组合

## What this test actually checks / 这个测试实际在检查什么

The consistency test is not a single metric. It checks multiple stages:

这个一致性测试不是单一指标，而是多阶段检查：

1. `weight_consistency`
   - whether mapped HF and Megatron weights match
2. `encoder_layer_wise`
   - whether encoder layer debug activations match layer by layer
3. `vision_encoder_layerwise`
   - whether key vision debug tensors match
4. `mllm_after_merger`
   - whether vision output after merger/adapter matches

1. `weight_consistency`
   - 映射后的 HF 和 Megatron 权重是否一致
2. `encoder_layer_wise`
   - encoder 层级 debug 激活是否逐层一致
3. `vision_encoder_layerwise`
   - vision 关键 debug 张量是否一致
4. `mllm_after_merger`
   - merger/adapter 之后的视觉输出是否一致

Do not interpret `overall_status` without checking which sub-test failed.

不要只看 `overall_status`，一定要看具体是哪个子测试失败。

## Recommended command pattern / 推荐命令模板

Run inside the container.

需要在容器里运行。

Set these first:

先设置这些环境变量：

```bash
export MODEL_NAME="llava-onevision2-4b"
export HF_MODEL_PATH="/path/to/hf_checkpoint"
export MCORE_CHECKPOINT_PREFIX="/path/to/mcore_checkpoint_prefix"
export TEST_IMAGE_PATH="/workspace/LLaVA-OneVision-2/asset/performance.png"
export TEST_PROFILE="low_vram"
export AIAK_TRAINING_PATH="/workspace/LLaVA-OneVision-2"
```

Then run one of:

然后运行以下之一：

```bash
export MASTER_PORT=26511; bash examples/llava_onevision2/check_model_consistency.sh 1 1
export MASTER_PORT=26512; bash examples/llava_onevision2/check_model_consistency.sh 2 1
export MASTER_PORT=26513; bash examples/llava_onevision2/check_model_consistency.sh 1 2
export MASTER_PORT=26514; bash examples/llava_onevision2/check_model_consistency.sh 2 2
```

The shell script resolves:

这个 shell 脚本会自动解析：

- `MCORE_CHECKPOINT_PATH=${MCORE_CHECKPOINT_PREFIX}_tp${TP}_pp${PP}`

So your checkpoint naming should match that convention.

所以你的 checkpoint 命名应符合这个约定。

## Checkpoint naming contract / checkpoint 命名约定

Expected paths:

期望路径：

- `..._tp1_pp1`
- `..._tp2_pp1`
- `..._tp1_pp2`
- `..._tp2_pp2`

Example:

例如：

```text
/foo/bar/iter_0001850_mcore_tp1_pp1
/foo/bar/iter_0001850_mcore_tp2_pp1
/foo/bar/iter_0001850_mcore_tp1_pp2
/foo/bar/iter_0001850_mcore_tp2_pp2
```

## Local image rule / 本地图像规则

Use a local image path.

请使用本地图像路径。

Do not rely on remote URLs in container runs.

不要在容器运行时依赖外链图片。

Recommended default:

推荐默认值：

```text
/workspace/LLaVA-OneVision-2/asset/performance.png
```

## How to read the result JSON / 怎么看结果 JSON

The result file is under:

结果文件位于：

```text
outputs/model_consistency_check/results_*.json
```

Read these in order:

按以下顺序看：

### 1. `overall_status`

This is only the final summary.

这只是最终汇总状态。

### 2. `tests.weight_consistency.status`

If this fails, first suspect:

如果这个失败，优先怀疑：

- wrong `MODEL_NAME`
- wrong TP/PP checkpoint path
- wrong weight mapping
- TP shard not gathered before compare

### 3. `tests.weight_consistency.weight_comparisons_summary`

Look at:

关注：

- `mismatched`
- `mismatches`

If most weights match and a few keys are `hf_key_not_found`, check whether they are expected structural differences.

如果大部分权重一致，只有少量 `hf_key_not_found`，先判断是不是预期中的结构差异。

### 4. `tests.vision_encoder_layerwise`

This checks a few strategic tensors:

这个测试关注几个关键张量：

- `after_patch_embed`
- `rotary_pos_emb`
- `after_pre_layernorm`
- `before_adapter`

If only `rotary_pos_emb` fails, verify that both sides are compared in the same debug representation.

如果只有 `rotary_pos_emb` 失败，先确认双方比较的是同一种 debug 表示。

### 5. `tests.encoder_layer_wise`

This is the strictest test.

这是最严格的测试。

Look at:

关注：

- `layer_comparisons_summary.mismatched_layers`
- `layer_comparisons_summary.mismatch_details`

If late layers fail while weight consistency passes, suspect debug capture semantics, tensor layout, or sequence-parallel / gather timing.

如果后面几层失败、但权重一致性通过，优先怀疑 debug 捕获语义、张量布局、或者 sequence-parallel / gather 时机。

### 6. `tests.mllm_after_merger.status`

If this passes, the end-of-vision pipeline is usually healthy.

如果这个通过，通常说明 vision pipeline 的最终输出链路基本健康。

## Practical interpretation / 实际判断顺序

Use this order when judging failures:

判断失败时建议按这个顺序：

1. `weight_consistency`
2. `vision_encoder_layerwise`
3. `mllm_after_merger`
4. `encoder_layer_wise`

Reason:

原因：

- weights failing usually means hard mismatch
- final merger failing usually means real pipeline inconsistency
- encoder-layer-wise failing can still be a debug-comparison issue rather than a true model bug

- 权重失败通常说明是硬错误
- merger 失败通常说明 pipeline 真有问题
- `encoder_layer_wise` 失败有时只是 debug 比较口径问题，不一定是模型真错

## Known repo-local lessons / 当前仓库已知经验

### 1. Rotary debug representation must be aligned

HF and Megatron may expose different `rotary_pos_emb` debug shapes.

HF 和 Megatron 可能暴露不同形状的 `rotary_pos_emb` debug 张量。

Example pattern:

例如：

- HF: `(1, S, 64)`
- Megatron: `(S, 32)`

Direct flatten+truncate comparison gives misleading cosine scores.

直接 flatten + truncate 比较会得到误导性的 cosine。

### 2. PP-aware testing is necessary

When `PP > 1`, not every pipeline stage owns:

当 `PP > 1` 时，不是每个 pipeline stage 都拥有：

- `vision_model`
- `adapter`
- decoder post-process outputs

So tests must skip non-owner stages instead of treating them as failures.

所以测试必须对非 owner stage 做 skip，而不是直接判失败。

### 3. TP-aware weight comparison is necessary

When `TP > 1`, do not compare a local TP shard directly against a full HF weight.

当 `TP > 1` 时，不能直接拿本地 TP shard 和完整 HF 权重比较。

Gather first along the correct dimension.

必须先沿正确维度 gather。

### 4. Encoder-layer-wise failures may still be debug-layout issues

If:

如果：

- weight consistency passes
- vision encoder strategic checks pass
- merger passes
- but `encoder_layer_wise` fails in later layers

then the issue may still be in debug tensor alignment or capture timing, not in the actual end-to-end model behavior.

那么问题仍有可能是 debug 张量对齐或捕获时机，而不是实际端到端模型行为错误。

## Minimal troubleshooting checklist / 最小排查清单

If the run fails, check in this order:

如果运行失败，按以下顺序排查：

1. Is `MODEL_NAME` correct for the checkpoint size?
2. Does `MCORE_CHECKPOINT_PREFIX` resolve to the right `*_tp${TP}_pp${PP}` directory?
3. Is `TEST_IMAGE_PATH` local and valid?
4. Does the container have enough GPUs for `TP * PP`?
5. Did `weight_consistency` fail because of real mismatches, or only a few expected missing keys?
6. Did a debug-only comparison fail due to layout or representation mismatch?

1. `MODEL_NAME` 是否与 checkpoint 尺寸匹配？
2. `MCORE_CHECKPOINT_PREFIX` 是否能正确解析到 `*_tp${TP}_pp${PP}`？
3. `TEST_IMAGE_PATH` 是否为有效本地路径？
4. 容器 GPU 数量是否满足 `TP * PP`？
5. `weight_consistency` 是真实大面积失败，还是只是少量预期缺 key？
6. 失败的是不是仅仅是 debug 比较口径问题？

## Expected output when using this skill / 使用这个 skill 时的期望输出

When asked to run or analyze a consistency test, return:

当被要求运行或分析一致性测试时，应返回：

1. the exact command used
2. the TP/PP combination
3. the result file path
4. the per-test statuses
5. the first failing test to investigate next

1. 实际执行命令
2. 使用的 TP/PP 组合
3. 结果文件路径
4. 各子测试状态
5. 下一步最值得继续排查的首个失败项
