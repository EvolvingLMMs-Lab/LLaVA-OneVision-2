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

There are two test systems in this repo:

本仓库有两套测试系统：

### 1. pytest test suite (recommended / 推荐)

- `tests/consistency/conftest.py` — session fixtures, HF→mcore conversion, Megatron initialization
- `tests/consistency/test_model_consistency.py` — 6 integration tests
- `tests/consistency/test_consistency_utils.py` — 10 utility functions + 11 unit tests
- `tests/consistency/run_consistency_tests.sh` — shell wrapper with auto-conversion + torchrun

- `tests/consistency/conftest.py` —— session 级 fixture、HF→mcore 转换、Megatron 初始化
- `tests/consistency/test_model_consistency.py` —— 6 个集成测试
- `tests/consistency/test_consistency_utils.py` —— 10 个工具函数 + 11 个单元测试
- `tests/consistency/run_consistency_tests.sh` —— shell 入口，自动转换 + torchrun

### 2. Legacy monolithic script (reference only / 仅供参考)

- `examples/llava_onevision2/check_model_consistency.sh`
- `examples/llava_onevision2/check_model_consistency.py`

仅作历史参考，新的工作请用 pytest 套件。

## Architecture / 架构

### Direction: HF → mcore

The pytest suite assumes **only the HF checkpoint exists** as input. The mcore checkpoint is **generated automatically** via conversion.

pytest 测试套件假设 **只有 HF checkpoint** 作为输入。mcore checkpoint 通过转换 **自动生成**。

```
HF auto-model (input)
  → convert_4b_hf_to_mcore.sh (auto-run by conftest.py or run_consistency_tests.sh)
    → mcore checkpoint (generated)
      → both models loaded → 6 tests run
```

### Direction: mcore → HF (reverse / deploy / round-trip) / 反向：mcore → HF（部署 / 回环）

The pytest suite **does not** exercise the reverse path. For the p14m2 variant,
two scripts ship for this:

| Script | Use case |
|---|---|
| `examples/llava_onevision2/convert/convert_4b_p14m2_mcore_to_hf.sh` | Single mcore→HF pass (deploy, inference debug) |
| `examples/llava_onevision2/convert/convert_4b_p14m2_mcore_to_release.sh` | Re-shard mcore via HF round-trip (change TP/PP without retraining) |

```bash
# mcore → HF (auto-detects /release subdir; pass either form)
bash examples/llava_onevision2/convert/convert_4b_p14m2_mcore_to_hf.sh \
    /train_tmp/llava_onevision2_4b_p14m2_mcore_tp1pp1 \
    /train_tmp/llava_onevision2_4b_p14m2_hf_out \
    1 1

# Re-shard: mcore TP=1 PP=1 → mcore TP=2 PP=4 (round-trips through HF)
bash examples/llava_onevision2/convert/convert_4b_p14m2_mcore_to_release.sh \
    /src_mcore_tp1pp1 /dst_mcore_tp2pp4 2 4 0,12,12,12
```

**Round-trip correctness** (TP=1 PP=1, verified 2026-05-25):
`mcore → HF → mcore` is **bitwise identical** to the original mcore checkpoint
(588 non-empty tensors compared, max abs diff = `0.000e+00`, 0 shape mismatches,
0 missing keys). This is the strongest correctness guarantee for the reverse
path. Use this whenever changing TP/PP layout without retraining.

pytest 套件 **不覆盖** 反向路径。p14m2 variant 提供两个脚本（单次 mcore→HF
用于部署/推理 debug，mcore→release 用于通过 HF 中转改 TP/PP 切分）。
回环 `mcore→HF→mcore` 在 TP=1 PP=1 下与原始 mcore 逐位一致（588 个非空 tensor，
max abs diff = `0.000e+00`，0 形状不匹配，0 缺失键，2026-05-25 验证）。
在不重训的前提下改 TP/PP layout 时使用回环。

> **Path convention note**: `convert_4b_p14m2_mcore_to_hf.sh` auto-detects
> `<load>/release` — pass either the parent dir or the explicit release path.
> Sibling scripts (4b, p14m3, p16m3, 8b, 30b) still require explicit `/release`.
>
> **路径约定说明**：`convert_4b_p14m2_mcore_to_hf.sh` 会自动检测
> `<load>/release` — 父目录或显式 release 路径都可以传。Sibling 脚本仍要求显式 `/release`。

### Test file structure / 测试文件结构

```
tests/consistency/
├── __init__.py                    # empty package init / 空包初始化
├── conftest.py                    # 9 session fixtures (209 lines) / 9 个 session fixture（209 行）
├── test_consistency_utils.py      # 10 utilities + 11 unit tests (373 lines, DO NOT MODIFY) / 10 个工具 + 11 个单元测试（373 行，不要改）
├── test_model_consistency.py      # 6 integration tests (402 lines) / 6 个集成测试（402 行）
└── run_consistency_tests.sh       # shell wrapper (60 lines) / shell 入口（60 行）
```

### Fixtures in conftest.py / conftest.py 中的 fixtures

下表列出 `conftest.py` 暴露的 session 级 fixture，及其用途和默认来源：

| Fixture | Scope | Description |
|---|---|---|
| `hf_model_path` | session | HF auto-model directory (env: `HF_MODEL_PATH`) |
| `converted_mcore_path` | session | Auto-converts HF→mcore if `MCORE_CHECKPOINT_PATH` not set |
| `preprocessor_path` | session | Processor path (defaults to `HF_MODEL_PATH`) |
| `test_image_path` | session | Local test image (default: `asset/performance.png`) |
| `megatron_init` | session | Initializes Megatron via sys.argv override |
| `hf_config` | session | `LlavaOnevision2Config.from_pretrained()` |
| `hf_vision_model` | session | `LlavaOnevision2Model.from_pretrained().visual` on cuda bf16 |
| `hf_cond_gen_model` | session | `LlavaOnevision2ForConditionalGeneration` on cuda bf16 |
| `mcore_model` | session | Megatron `get_model()` + `load_checkpoint()` |
| `hf_processor` | session | `AutoProcessor.from_pretrained()` |

## What the 6 tests check / 6 个测试检查什么

### test_weight_consistency (fast)

Compares all mapped weights between HF and mcore vision models:

比较 HF 和 mcore 视觉模型之间所有映射权重：

- Patch embedding (conv weight + bias) / patch embedding（卷积 weight + bias）
- Class embedding / class embedding
- Pre/post layer norms / 前/后 layer norm
- Per-layer (24 layers): QKV weight/bias, projection, MLP fc1/fc2, layer norms / 每层（24 层）：QKV weight/bias、projection、MLP fc1/fc2、layer norms
- QKV layout conversion via `convert_hf_qkv_to_mcore_layout` (interleaved Q/K/V per head) / 通过 `convert_hf_qkv_to_mcore_layout` 做 QKV 布局转换（每 head 交织 Q/K/V）
- TP-aware gathering via `_maybe_gather_tp_weight` / 通过 `_maybe_gather_tp_weight` 做 TP-aware gather
- Threshold: cosine > 0.9999 / 阈值：cosine > 0.9999

### test_vision_encoder_consistency_336px (fast)

Compares `forward_debug` outputs at 4 strategic points:

在 4 个关键点比较 `forward_debug` 输出：

- `after_patch_embed` — patch embedding output / patch embedding 输出
- `rotary_pos_emb` — rotary position embedding (aligned via `align_rotary_debug_tensors`) / 旋转位置编码（通过 `align_rotary_debug_tensors` 对齐）
- `after_pre_layernorm` — after pre-layernorm / 经过 pre-layernorm 之后
- `before_adapter` — final encoder output before adapter / 进入 adapter 之前的最终 encoder 输出
- Threshold: cosine > 0.99 / 阈值：cosine > 0.99

### test_mllm_after_merger_336px (fast)

Compares vision + adapter pipeline output:

比较视觉 + adapter pipeline 输出：

- HF: `forward_debug['after_merger']` / HF：`forward_debug['after_merger']`
- mcore: `vision_model()` → `adapter()` / mcore：`vision_model()` → `adapter()`
- Threshold: cosine > 0.99 / 阈值：cosine > 0.99

### test_encoder_layer_wise_consistency (slow)

Layer-by-layer comparison of all 24 encoder layers:

逐层比较所有 24 个 encoder 层：

- `layer_{i}_input` and `layer_{i}_output` for each layer / 每层的 `layer_{i}_input` 和 `layer_{i}_output`
- `input_hidden_states` — initial encoder input / 初始 encoder 输入
- `final_output` — final encoder output / 最终 encoder 输出
- Uses `align_encoder_debug_tensors` for shape alignment / 用 `align_encoder_debug_tensors` 做形状对齐
- Threshold: cosine > 0.99 / 阈值：cosine > 0.99

### test_llm_output_consistency (slow)

End-to-end LLM logits comparison:

端到端 LLM logits 比较：

- Loads `LlavaOnevision2ForConditionalGeneration` (HF) and full mcore model / 加载 HF 的 `LlavaOnevision2ForConditionalGeneration` 和完整 mcore 模型
- Tokenizes prompt with image, runs forward pass on both / 用图像 tokenize prompt，两边都跑 forward
- Compares output logits / 比较输出 logits
- Threshold: cosine > 0.99 / 阈值：cosine > 0.99

### test_hf_loading_consistency (slow)

Validates HF model loading methods are equivalent:

验证 HF 模型加载方式等价：

- `from_pretrained()` vs manual `load_file()` from safetensors / `from_pretrained()` 对比从 safetensors 手动 `load_file()`
- Compares all vision weights (exact match via `np.allclose`) / 比较所有 vision 权重（用 `np.allclose` 做精确匹配）
- Compares `forward_debug` outputs (cosine > 0.9999) / 比较 `forward_debug` 输出（cosine > 0.9999）

## Environment variables / 环境变量

| Variable | Default | Description |
|---|---|---|
| `HF_MODEL_PATH` | `<path/to/hf_checkpoint>` | HF checkpoint (the only required input) |
| `MCORE_CHECKPOINT_PATH` | (auto-generated) | Set to skip conversion |
| `PREPROCESSOR_PATH` | `$HF_MODEL_PATH` | Image processor path |
| `TEST_IMAGE_PATH` | `$REPO_ROOT/asset/performance.png` | Local test image |
| `CONSISTENCY_TEST_TP` | `1` | Tensor parallel size |
| `CONSISTENCY_TEST_PP` | `1` | Pipeline parallel size |
| `AIAK_TRAINING_PATH` | `$REPO_ROOT` | AIAK training framework root |
| `AIAK_MAGATRON_PATH` | `$REPO_ROOT/aiak_megatron` | AIAK Megatron path |
| `MASTER_PORT` | `29500` | Distributed master port |

## How to run / 怎么跑

All Python must run inside the container `llava_megatron_container_ax`.

所有 Python 必须在容器 `llava_megatron_container_ax` 内运行。

### Quick: run non-slow tests with auto-conversion / 快速：跑非 slow 测试 + 自动转换

```bash
# Inside container, from repo root:
# 在容器内、仓库根目录执行：
bash tests/consistency/run_consistency_tests.sh
```

### Run all tests including slow / 跑全部测试（含 slow）

```bash
bash tests/consistency/run_consistency_tests.sh -m ""
```

### Custom TP/PP / 自定义 TP/PP

```bash
TP=2 PP=1 MASTER_PORT=29501 bash tests/consistency/run_consistency_tests.sh
```

### Skip conversion (pre-existing mcore checkpoint) / 跳过转换（已有 mcore checkpoint）

```bash
MCORE_CHECKPOINT_PATH=/path/to/existing bash tests/consistency/run_consistency_tests.sh
```

### Run only unit tests (no GPU needed, works on host) / 只跑单元测试（不需要 GPU，host 上也能跑）

```bash
pytest tests/consistency/test_consistency_utils.py -v
```

### Run specific integration test / 跑指定的集成测试

```bash
bash tests/consistency/run_consistency_tests.sh -k test_weight_consistency
```

## What run_consistency_tests.sh does / run_consistency_tests.sh 做了什么

1. Validates `HF_MODEL_PATH` and `TEST_IMAGE_PATH` exist / 校验 `HF_MODEL_PATH` 和 `TEST_IMAGE_PATH` 存在
2. If `MCORE_CHECKPOINT_PATH` is empty, runs `convert_4b_hf_to_mcore.sh` to generate it / 如果 `MCORE_CHECKPOINT_PATH` 为空，跑 `convert_4b_hf_to_mcore.sh` 生成
3. Exports all env vars for conftest.py / 为 conftest.py 导出所有环境变量
4. Sets `PYTHONPATH` to include `transformers_impl/llavaonevision2`, `aiak_megatron`, repo root / 把 `transformers_impl/llavaonevision2`、`aiak_megatron`、仓库根目录加入 `PYTHONPATH`
5. Launches `torchrun --nproc_per_node=$((TP*PP))` with pytest / 用 `torchrun --nproc_per_node=$((TP*PP))` 启动 pytest

## What conftest.py does for Megatron init / conftest.py 如何初始化 Megatron

Since pytest has its own arg parsing, Megatron CLI args can't be passed via command line. The solution:

由于 pytest 有自己的参数解析，Megatron CLI 参数不能通过命令行传递。解决方案：

1. Shell script exports env vars (`HF_MODEL_PATH`, `MCORE_CHECKPOINT_PATH`, `CONSISTENCY_TEST_TP/PP`, etc.) / shell 脚本导出环境变量（`HF_MODEL_PATH`、`MCORE_CHECKPOINT_PATH`、`CONSISTENCY_TEST_TP/PP` 等）
2. `conftest.py` reads env vars, temporarily overrides `sys.argv` with constructed Megatron CLI args / `conftest.py` 读取环境变量，临时把 `sys.argv` 替换成构造好的 Megatron CLI 参数
3. Calls `parse_arguments()` + `initialize_aiak_megatron()` inside the override / 在替换期内调用 `parse_arguments()` + `initialize_aiak_megatron()`
4. Restores `sys.argv` afterward / 完事后恢复 `sys.argv`

## How to interpret failures / 如何解读失败

### Priority order for diagnosis / 诊断优先顺序

1. **test_weight_consistency** — If this fails, all other tests are unreliable / 这个挂了，其他测试都不可信
2. **test_vision_encoder_consistency_336px** — Strategic checkpoint comparison / 关键 checkpoint 点比较
3. **test_mllm_after_merger_336px** — Vision + adapter pipeline health / 视觉 + adapter pipeline 健康度
4. **test_encoder_layer_wise_consistency** — May fail due to debug alignment, not real bugs / 可能因 debug 对齐问题失败，未必是真 bug
5. **test_llm_output_consistency** — Full end-to-end, most sensitive to any discrepancy / 完整端到端，对任何偏差最敏感
6. **test_hf_loading_consistency** — HF-only test, independent of mcore / 仅 HF 的测试，与 mcore 无关

### Common failure causes / 常见失败原因

| Symptom | Likely Cause | Fix |
|---|---|---|
| weight_consistency fails on QKV | QKV layout conversion bug | Check `convert_hf_qkv_to_mcore_layout` for num_heads |
| weight_consistency fails on many keys | Wrong model / TP/PP mismatch | Verify `HF_MODEL_PATH` and conversion TP/PP |
| vision_encoder rotary_pos_emb fails | Debug tensor shape mismatch | Check `align_rotary_debug_tensors` — HF `(1,S,64)` vs mcore `(S,32)` |
| encoder_layer_wise late layers fail | Debug capture timing / layout | Usually not a real model bug if weight + merger pass |
| llm_output shape mismatch | Wrong tokenization or attention mask | Check prompt formatting and `attention_mask.logical_not()` |
| Megatron init fails | Wrong CLI args | Check `_build_megatron_cli_args` in conftest.py |
| Conversion fails | Missing `AIAK_TRAINING_PATH` | Export it before running |

### Key weight mapping / 关键权重映射

| HF Key | mcore Key |
|---|---|
| `embeddings.patch_embedding` | `patch_embed.proj` |
| `embeddings.class_embedding` | `class_embedding` |
| `layernorm_pre/post` | `pre_layernorm/post_layernorm` |
| `encoder.layers.{i}.layer_norm1` | `decoder.layers.{i}.self_attention.linear_qkv.layer_norm` |
| `encoder.layers.{i}.self_attn.qkv` | `decoder.layers.{i}.self_attention.linear_qkv` |
| `encoder.layers.{i}.self_attn.proj` | `decoder.layers.{i}.self_attention.linear_proj` |
| `encoder.layers.{i}.layer_norm2` | `decoder.layers.{i}.mlp.linear_fc1.layer_norm` |
| `encoder.layers.{i}.mlp.fc1/fc2` | `decoder.layers.{i}.mlp.linear_fc1/fc2` |

QKV weights need layout conversion: HF stores `[Q_all, K_all, V_all]`, mcore stores interleaved `[Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]`.

QKV 权重需要布局转换：HF 存储 `[Q_all, K_all, V_all]`，mcore 存储交织的 `[Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]`。

## Known repo-local lessons / 当前仓库已知经验

### 1. Rotary debug representation must be aligned

HF and Megatron expose different `rotary_pos_emb` debug shapes:

HF 和 Megatron 暴露不同形状的 `rotary_pos_emb` debug 张量：

- HF: `(1, S, 64)`
- Megatron: `(S, 32)`

The `align_rotary_debug_tensors` function handles this by squeezing batch dim and concatenating mcore's half-dim.

`align_rotary_debug_tensors` 函数通过去掉 batch 维度并拼接 mcore 的半维度来处理。

### 2. PP-aware testing is necessary

When `PP > 1`, not every pipeline stage owns `vision_model`, `adapter`, or decoder post-process outputs. Tests must skip non-owner stages.

当 `PP > 1` 时，不是每个 pipeline stage 都拥有 `vision_model`、`adapter` 或 decoder 后处理输出。测试必须跳过非 owner stage。

### 3. TP-aware weight comparison is necessary

When `TP > 1`, use `_maybe_gather_tp_weight` to gather shards before comparison. It gathers along first dim for QKV/FC1, last dim for proj/FC2.

当 `TP > 1` 时，用 `_maybe_gather_tp_weight` 在比较前 gather shards。QKV/FC1 沿第一维 gather，proj/FC2 沿最后一维。

### 4. HF and mcore use the same pixel value 2x2 memory layout

No pixel value conversion is needed between HF and mcore models.

HF 和 mcore 模型使用相同的 2x2 内存布局，无需转换 pixel values。

### 5. Encoder-layer-wise failures may be debug-layout issues

If weight_consistency + merger pass but encoder_layer_wise fails in late layers, suspect debug capture semantics rather than real model bugs.

如果 weight_consistency + merger 通过但 encoder_layer_wise 在后面层失败，优先怀疑 debug 捕获语义而非模型真错。

## Minimal troubleshooting checklist / 最小排查清单

If the run fails, check in this order:

如果运行失败，按以下顺序排查：

1. Is the container running? `docker exec -it llava_megatron_container_ax bash`
2. Does `HF_MODEL_PATH` exist and contain safetensors files?
3. Did the HF→mcore conversion succeed? Check stderr output.
4. Does the container have enough GPUs for `TP * PP`?
5. Is `MASTER_PORT` already in use? Try a different port.
6. Did `test_weight_consistency` fail? → Fix this first before investigating other tests.
7. Is the failure in a `@pytest.mark.slow` test? → Run fast tests first with default marker filter.

1. 容器是否在运行？`docker exec -it llava_megatron_container_ax bash`
2. `HF_MODEL_PATH` 是否存在且包含 safetensors 文件？
3. HF→mcore 转换是否成功？检查 stderr 输出。
4. 容器 GPU 数量是否满足 `TP * PP`？
5. `MASTER_PORT` 是否被占用？换一个端口试试。
6. `test_weight_consistency` 是否失败？→ 先修这个再看其他测试。
7. 失败的是否是 `@pytest.mark.slow` 测试？→ 先用默认 marker 跑 fast 测试。
