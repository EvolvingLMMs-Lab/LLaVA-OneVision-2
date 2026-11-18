# `transformers_impl/` — HuggingFace-format Tooling

> Bilingual / 中英双语. English first, then 中文.

This directory hosts the HuggingFace-side tooling for LLaVA-OneVision-2.5:
modeling files (`llavaonevision1_5/`, `llavaonevision2/`), the unified merge
package `merge_ov2/` that fuses **ViT + Adapter + LLM** into a single
`AutoModelForCausalLM`-loadable checkpoint (dense **and** MoE), plus inference
helpers (`inference.py`, `dist_run.py`).

The two top-level scripts `merge_ov2.py` and `merge_ov2_moe.py` are
backward-compat shims (9 lines each) that forward to `merge_ov2.cli` with the
appropriate `--variant`. New code should call the package directly.

---

## English

### What `merge_ov2/` does

Given three independently-pretrained components:

- **ViT encoder** (e.g. `/ov2/pretrain_models/onevision-encoder-large`)
- **Adapter** (optional — initialized fresh if omitted)
- **LLM** (dense Qwen3 / Qwen2.5, or MoE Qwen3-30B-A3B)

…it produces a single HuggingFace checkpoint with the modeling code copied
in-tree (`trust_remote_code=True`-loadable). bf16 by default, byte-equal to
the legacy single-file scripts it replaces, and built with **strict**
in-place loading + streaming safetensors I/O (no full `state_dict` copy in
RAM).

### Three subcommands

```
PYTHONPATH=transformers_impl:. python -m merge_ov2 {merge,validate,dry-run} ...
```

> **Path note**: `merge_ov2` is the **package** under `transformers_impl/merge_ov2/`.
> The two top-level files `transformers_impl/merge_ov2.py` and
> `transformers_impl/merge_ov2_moe.py` are 9-line BC shims and **cannot** be
> invoked as `python -m transformers_impl.merge_ov2` (Python resolves to the
> shim file, not the package). Always set `PYTHONPATH=transformers_impl:.`
> and call `python -m merge_ov2 ...` directly.

- **`merge`** — full pipeline: build empty model → load weights (strict, in-place)
  → run the configured validators → save in target dtype.
- **`validate`** — load an already-merged checkpoint and rerun the validators
  (useful after manual surgery or to confirm a downloaded checkpoint).
- **`dry-run`** — remap weights and report load coverage (missing /
  unexpected / shape-mismatch keys), do **not** save. Cheap and fast — use
  this first when wiring up a new ViT/LLM combo.

### Quick start (dense)

```bash
docker exec llava_megatron_container_ax bash -lc \
  "cd /workspace/LLaVA-OneVision-2.5 && PYTHONPATH=transformers_impl:. \
   python -m merge_ov2 merge \
     --variant dense \
     --vit       /ov2/pretrain_models/onevision-encoder-large \
     --llm       /ov2/pretrain_models/Qwen3-1.7B-Base \
     --processor /ov2/pretrain_models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
     --out       /tmp/ov2-dense-merged \
     --target-dtype bf16 \
     --img         https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
     --sample-text 'Hello, my dog is cute'"
```

### Quick start (MoE)

```bash
PYTHONPATH=transformers_impl:. python -m merge_ov2 merge \
  --variant moe \
  --vit       /ov2/pretrain_models/onevision-encoder-large \
  --llm       /mnt/publicdataset/Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --processor /ov2/pretrain_models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --out       /tmp/ov2-moe-merged \
  --target-dtype bf16 \
  --img         https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
  --sample-text 'Hello, my dog is cute'
```

### OV2 4B variants — parameter cheat sheet

The dense variant covers two real-world OV2 4B configurations. Always pass the
matching `--patch-size` (via the ViT checkpoint) and `--spatial-merge-size`,
plus the right `--vit-validator-strategy` — using the wrong validator silently
crashes inside reshape ops.

| Variant tag | ViT checkpoint suffix | `--spatial-merge-size` | `--vit-validator-strategy` | Effective `image_size` |
|---|---|---|---|---|
| `4b` (legacy) | `onevision-encoder-large` (patch14) | `2` | `blockorder` (default) or `layerwise` | 14 × 2 × N |
| `4b_p16m3` (current) | `onevision_encoder_patch16_*` | `3` | `layerwise` (**required**) | 16 × 3 × N = 48 × N |

> **Why `layerwise` is required for `4b_p16m3`**: the `blockorder` validator
> hard-codes `patch_size=14, spatial_merge_size=2` in its reshape; with
> `patch_size=16, spatial_merge_size=3` the reshape dimensions don't divide
> evenly and you get a cryptic `RuntimeError: shape '[...]' is invalid for input of size N`.
> Pass `--vit-validator-strategy layerwise` to bypass this entirely.

### Skipping validators

`--validate-skip` is **append**-style — pass it once per validator you want to
skip. Three validators exist: `vit`, `llm`, `e2e`.

```bash
# Skip only the e2e validator (run vit + llm)
... --validate-skip e2e

# Validate only LLM (skip vit + e2e — useful when ViT is huge or unchanged)
... --validate-skip vit --validate-skip e2e

# Skip everything (fast merge, no parity check; do this only after a known-good run)
... --validate-skip vit --validate-skip llm --validate-skip e2e
```

If you skip `vit` you may also omit `--qwen-processor` and `--img`; if you
skip `llm` you may omit `--sample-text`. The CLI enforces only the required
flags for the validators you actually run.

### Post-merge: convert to Megatron + run consistency tests

After a successful merge, the standard next step is to convert the HF
checkpoint to Megatron-Core format and verify HF↔mcore parity end-to-end.
That workflow is documented in the `llava-onevision2-consistency` skill —
load it with `skill(name="llava-onevision2-consistency")` for the full
recipe (TP/PP layouts, threshold table, runner script).

Quick pointer:

```bash
# 1) HF → mcore (TP=1 PP=1)
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2.5 \
bash examples/llava_onevision2/convert/convert_4b_hf_to_mcore.sh \
  /train_tmp/ov2-dense-merged \
  /train_tmp/ov2-dense-merged_mcore_tp1_pp1 \
  1 1

# 2) Run consistency tests (6 checks, see consistency skill for thresholds)
docker exec llava_megatron_container_ax bash -c "
  cd /workspace/LLaVA-OneVision-2.5 && \
  HF_MODEL_PATH=/train_tmp/ov2-dense-merged \
  MCORE_CKPT_PATH=/train_tmp/ov2-dense-merged_mcore_tp1_pp1 \
  MODEL_VARIANT=4b_p16m3 \
  bash tests/consistency/run_consistency_tests.sh -v --tb=short
"
```

### Dry-run before committing GPU time

```bash
PYTHONPATH=transformers_impl:. python -m merge_ov2 dry-run \
  --variant moe \
  --vit /path/to/vit --llm /path/to/llm \
  --processor /path/to/processor
```

Exits non-zero if any tensor is missing, unexpected, or shape-mismatched.
This is the fastest way to verify a new ViT/LLM pairing without spending GPU
hours on a doomed merge.

### Validator strategies

| Flag | Default (dense) | Default (MoE) | Notes |
|---|---|---|---|
| `--vit-validator-strategy` | `blockorder` | `layerwise` | `layerwise` does per-block cosine; `blockorder` does whole-ViT forward. |
| `--llm-validator-strategy` | `parallel` | `sequential` | `parallel` loads original LLM alongside merged → OOMs on 30B MoE. CLI rejects `parallel + moe` with exit 2. |

`--qwen-processor` is **optional** — it defaults to `--processor`. The ViT
validator reloads it as a fresh `Qwen2VLImageProcessor` instance and locally
overrides `do_resize=False`, `max_pixels=min_pixels=h*w`, etc. Since reload
goes through `from_pretrained`, the production processor saved with the
checkpoint is never mutated. Pass an explicit `--qwen-processor` only if
you want a different reference processor for validation (e.g. a vanilla
Qwen2.5-VL one for cross-checking).

### Loading the merged checkpoint

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "/tmp/ov2-dense-merged",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("/tmp/ov2-dense-merged", trust_remote_code=True)
```

The modeling files (`modeling_llava_onevision2*.py`,
`configuration_llava_onevision2*.py`) are copied into the output directory
alongside the safetensors shards, so the checkpoint is self-contained.

---

### Testing

All tests run in the Docker container `llava_megatron_container_ax`.
The `tests/merge_ov2/` tree is organized by purpose:

```
tests/merge_ov2/
├── cli/         # argparse contract, no-old-imports gate
├── dense/       # dense dry-run + corruption tests
├── moe/         # MoE dry-run, corruption, validator dispatch
├── validators/  # validate-skip + validate-subcommand e2e (F2-gated)
└── quality/     # dead-code gate (Phase 8 A10)
```

Plus `tests/consistency/` (HF↔Megatron parity) and `tests/perf/`
(memory-measurement shell scripts).

#### 1. Always-on tests (no real checkpoints needed)

These build a tiny synthetic ViT+adapter+LLM triplet on the fly via
`tests/fixtures/build_tiny_triplet.py`, run the merge, and check the result.

```bash
docker exec llava_megatron_container_ax bash -lc \
  "cd /workspace/LLaVA-OneVision-2.5 && PYTHONPATH=transformers_impl:. \
   pytest tests/merge_ov2/ -v"
```

Expected: ~14 pass, ~6 skip (those that need real F2 fixtures).

#### 2. Dead-code gate (Phase 8 A10)

```bash
pytest tests/merge_ov2/quality/test_no_dead_helpers.py -v
# or run the underlying script directly:
python tests/merge_ov2/quality/check_no_dead_helpers.py
```

Asserts the new package contains no legacy helpers
(`convert_block_to_rowmajor_layout`, `create_test_image`, numpy cosine
helper, `load_state_dict(strict=False)`).

#### 3. Real-checkpoint tests (opt-in)

These need the real F2 fixtures listed below and are gated by env vars.

| Env var | Enables |
|---|---|
| `OV2_REAL_FIXTURE=1` | F2-dense tests (validators, validate-subcommand e2e, etc.) |
| `OV2_REAL_FIXTURE_MOE=1` | additionally, F2-moe tests (default-strategy dispatch, override dispatch) |
| `OV2_F2_PROCESSOR=<path>` | override the processor path |

Local fixture paths the suite expects (override via env if your layout differs):

- ViT: `/ov2/pretrain_models/onevision-encoder-large`
- Dense LLM: `/ov2/pretrain_models/Qwen3-1.7B-Base`
- MoE LLM: `/mnt/publicdataset/Qwen/Qwen3-30B-A3B-Instruct-2507`
- Processor: `/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`
- Qwen processor: `/ov2/pretrain_models/Qwen2.5-VL-7B-Instruct-processor`

Example — run dense F2 validator tests:

```bash
docker exec llava_megatron_container_ax bash -lc \
  "cd /workspace/LLaVA-OneVision-2.5 && PYTHONPATH=transformers_impl:. \
   OV2_REAL_FIXTURE=1 \
   pytest tests/merge_ov2/validators/ -v"
```

Example — full MoE F2 dispatch suite (needs an 80 GB GPU for the 30B model):

```bash
docker exec llava_megatron_container_ax bash -lc \
  "cd /workspace/LLaVA-OneVision-2.5 && PYTHONPATH=transformers_impl:. \
   OV2_REAL_FIXTURE=1 OV2_REAL_FIXTURE_MOE=1 \
   pytest tests/merge_ov2/moe/test_validators.py -v"
```

#### 4. Memory measurement helpers

```bash
bash tests/perf/measure_rss.sh       # peak host RSS during dense merge
bash tests/perf/measure_gpu_mem.sh   # peak GPU memory during dense merge
```

Both auto-skip without `OV2_REAL_FIXTURE=1`.

---

## 中文

### `merge_ov2/` 是做什么的

把三份独立预训练好的组件：

- **ViT 编码器**（如 `/ov2/pretrain_models/onevision-encoder-large`）
- **Adapter**（可选；不传就用默认初始化）
- **LLM**（dense 的 Qwen3 / Qwen2.5，或 MoE 的 Qwen3-30B-A3B）

合并成一份 HuggingFace 格式的 checkpoint，可以用 `AutoModelForCausalLM` +
`trust_remote_code=True` 直接加载。默认 bf16，与被替换的旧脚本 byte-equal，
内部使用**严格的**就地加载 + 流式 safetensors I/O（不会在内存里复制完整
`state_dict`）。

### 三个子命令

```
PYTHONPATH=transformers_impl:. python -m merge_ov2 {merge,validate,dry-run} ...
```

> **路径说明**：`merge_ov2` 是 `transformers_impl/merge_ov2/` 这个**包**。
> 顶层的 `transformers_impl/merge_ov2.py` 和 `transformers_impl/merge_ov2_moe.py`
> 是 9 行的 BC shim 脚本，**不能**用 `python -m transformers_impl.merge_ov2`
> 调用（Python 会解析到 shim 文件而不是包）。一律用
> `PYTHONPATH=transformers_impl:. python -m merge_ov2 ...`。

- **`merge`** —— 全流程：构建空模型 → 严格就地加载权重 → 跑校验器 → 按目标 dtype 保存。
- **`validate`** —— 加载一个已合并好的 checkpoint，再跑一遍校验器（手动改过权重、或验证下载来的 checkpoint 时有用）。
- **`dry-run`** —— 只做权重 remap + 输出 load 覆盖率报告（missing / unexpected / shape-mismatch），**不**保存。
  搭新的 ViT/LLM 组合时先跑这个，便宜又快。

### 快速上手（dense）

```bash
docker exec llava_megatron_container_ax bash -lc \
  "cd /workspace/LLaVA-OneVision-2.5 && PYTHONPATH=transformers_impl:. \
   python -m merge_ov2 merge \
     --variant dense \
     --vit       /ov2/pretrain_models/onevision-encoder-large \
     --llm       /ov2/pretrain_models/Qwen3-1.7B-Base \
     --processor /ov2/pretrain_models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
     --out       /tmp/ov2-dense-merged \
     --target-dtype bf16 \
     --img         https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
     --sample-text 'Hello, my dog is cute'"
```

### 快速上手（MoE）

```bash
PYTHONPATH=transformers_impl:. python -m merge_ov2 merge \
  --variant moe \
  --vit       /ov2/pretrain_models/onevision-encoder-large \
  --llm       /mnt/publicdataset/Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --processor /ov2/pretrain_models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --out       /tmp/ov2-moe-merged \
  --target-dtype bf16 \
  --img         https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg \
  --sample-text 'Hello, my dog is cute'
```

### OV2 4B 各 variant 参数对照表

dense variant 涵盖了两套真实使用的 OV2 4B 配置。`--patch-size`（由 ViT
checkpoint 决定）、`--spatial-merge-size` 和 `--vit-validator-strategy` 三者
必须配齐 —— 用错 validator 会在 reshape 里直接崩。

| Variant tag | ViT 后缀 | `--spatial-merge-size` | `--vit-validator-strategy` | 等效 `image_size` |
|---|---|---|---|---|
| `4b`（旧） | `onevision-encoder-large`（patch14） | `2` | `blockorder`（默认）或 `layerwise` | 14 × 2 × N |
| `4b_p16m3`（当前） | `onevision_encoder_patch16_*` | `3` | `layerwise`（**必须**） | 16 × 3 × N = 48 × N |

> **`4b_p16m3` 为什么必须用 `layerwise`**：`blockorder` validator 的 reshape
> 写死了 `patch_size=14, spatial_merge_size=2` 的尺寸假设；换成
> `patch_size=16, spatial_merge_size=3` 后 reshape 维度对不上，会抛
> `RuntimeError: shape '[...]' is invalid for input of size N`。
> 显式 `--vit-validator-strategy layerwise` 直接绕开。

### 跳过 validator

`--validate-skip` 是 **append** 模式，每跳一个就传一次。一共三个 validator：
`vit`、`llm`、`e2e`。

```bash
# 只跳 e2e（跑 vit + llm）
... --validate-skip e2e

# 只验证 LLM（跳过 vit + e2e —— ViT 太大或没改动时常用）
... --validate-skip vit --validate-skip e2e

# 全跳过（最快 merge，无任何校验；只在已知配置稳定时才这样跑）
... --validate-skip vit --validate-skip llm --validate-skip e2e
```

跳了 `vit` 时也可以省略 `--qwen-processor` 和 `--img`；跳了 `llm` 时可以
省略 `--sample-text`。CLI 只对你实际要跑的 validator 强制要求对应的参数。

### Merge 完之后：转 Megatron + 跑一致性测试

merge 成功之后，标准下一步是把 HF checkpoint 转成 Megatron-Core 格式，再跑
HF↔mcore 的端到端一致性验证。这套流程在 `llava-onevision2-consistency` skill
里 —— 用 `skill(name="llava-onevision2-consistency")` 加载，里面有 TP/PP
布局、阈值表、runner 脚本完整说明。

简版：

```bash
# 1) HF → mcore (TP=1 PP=1)
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2.5 \
bash examples/llava_onevision2/convert/convert_4b_hf_to_mcore.sh \
  /train_tmp/ov2-dense-merged \
  /train_tmp/ov2-dense-merged_mcore_tp1_pp1 \
  1 1

# 2) 跑一致性测试（6 个 check，阈值见 consistency skill）
docker exec llava_megatron_container_ax bash -c "
  cd /workspace/LLaVA-OneVision-2.5 && \
  HF_MODEL_PATH=/train_tmp/ov2-dense-merged \
  MCORE_CKPT_PATH=/train_tmp/ov2-dense-merged_mcore_tp1_pp1 \
  MODEL_VARIANT=4b_p16m3 \
  bash tests/consistency/run_consistency_tests.sh -v --tb=short
"
```

### 在花 GPU 时间之前先 dry-run

```bash
PYTHONPATH=transformers_impl:. python -m merge_ov2 dry-run \
  --variant moe \
  --vit /path/to/vit --llm /path/to/llm \
  --processor /path/to/processor
```

发现任何 missing / unexpected / shape-mismatch 立刻非零退出。
新搭 ViT/LLM 组合时这是最快的体检。

### 校验器策略

| 参数 | dense 默认 | MoE 默认 | 说明 |
|---|---|---|---|
| `--vit-validator-strategy` | `blockorder` | `layerwise` | `layerwise` 逐 block 比 cosine；`blockorder` 整体 ViT forward。 |
| `--llm-validator-strategy` | `parallel` | `sequential` | `parallel` 把原 LLM 和合并后模型同时加载 → 30B MoE 必爆显存。CLI 在 `parallel + moe` 时直接 exit 2。 |

`--qwen-processor` 是**可选**的，不传就默认等于 `--processor`。ViT
校验器内部会用 `from_pretrained` 重新 load 一份独立的 `Qwen2VLImageProcessor`
实例，再在本地把 `do_resize=False`、`max_pixels=min_pixels=h*w` 等字段覆写掉
—— 因为是重新 load 出来的新实例，不会污染最终保存到 checkpoint 里的那份
production processor。只有当你确实想用另一份 processor 当校验对照
（比如 vanilla Qwen2.5-VL）时才需要显式传 `--qwen-processor`。

### 加载合并后的 checkpoint

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "/tmp/ov2-dense-merged",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained("/tmp/ov2-dense-merged", trust_remote_code=True)
```

modeling 文件（`modeling_llava_onevision2*.py`、`configuration_llava_onevision2*.py`）
会跟着 safetensors 一起拷到输出目录，checkpoint 是自包含的。

---

### 测试

所有测试都在 Docker 容器 `llava_megatron_container_ax` 里跑。
`tests/merge_ov2/` 按用途分目录组织：

```
tests/merge_ov2/
├── cli/         # argparse 契约、no-old-imports 闸
├── dense/       # dense 的 dry-run + 损坏测试
├── moe/         # MoE 的 dry-run、损坏、validator dispatch
├── validators/  # validate-skip + validate 子命令 e2e（F2 才跑）
└── quality/     # 死代码闸（Phase 8 A10）
```

另外还有 `tests/consistency/`（HF↔Megatron 一致性）和 `tests/perf/`
（显存 / 内存测量脚本）。

#### 1. 不依赖真权重的测试（始终能跑）

通过 `tests/fixtures/build_tiny_triplet.py` 现场造一份 tiny ViT+adapter+LLM
三元组，跑完整 merge，再断言结果。

```bash
docker exec llava_megatron_container_ax bash -lc \
  "cd /workspace/LLaVA-OneVision-2.5 && PYTHONPATH=transformers_impl:. \
   pytest tests/merge_ov2/ -v"
```

预期：约 14 个 pass，约 6 个 skip（需要真 F2 fixture 的部分）。

#### 2. 死代码闸（Phase 8 A10）

```bash
pytest tests/merge_ov2/quality/test_no_dead_helpers.py -v
# 或直接跑底层脚本：
python tests/merge_ov2/quality/check_no_dead_helpers.py
```

断言新包里不再含旧 helper（`convert_block_to_rowmajor_layout`、
`create_test_image`、numpy cosine helper、`load_state_dict(strict=False)`）。

#### 3. 真权重测试（opt-in）

需要本地有下面的 F2 fixture，且通过环境变量打开：

| 环境变量 | 打开什么 |
|---|---|
| `OV2_REAL_FIXTURE=1` | F2-dense 系列（校验器、validate 子命令端到端等） |
| `OV2_REAL_FIXTURE_MOE=1` | 在前一项基础上，再打开 F2-moe（默认策略 dispatch、override dispatch） |
| `OV2_F2_PROCESSOR=<path>` | 覆写 processor 路径 |

本地 fixture 路径（如果你的本地布局不同，用环境变量覆写）：

- ViT：`/ov2/pretrain_models/onevision-encoder-large`
- Dense LLM：`/ov2/pretrain_models/Qwen3-1.7B-Base`
- MoE LLM：`/mnt/publicdataset/Qwen/Qwen3-30B-A3B-Instruct-2507`
- Processor：`/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`
- Qwen processor：`/ov2/pretrain_models/Qwen2.5-VL-7B-Instruct-processor`

示例 —— 跑 dense F2 校验器测试：

```bash
docker exec llava_megatron_container_ax bash -lc \
  "cd /workspace/LLaVA-OneVision-2.5 && PYTHONPATH=transformers_impl:. \
   OV2_REAL_FIXTURE=1 \
   pytest tests/merge_ov2/validators/ -v"
```

示例 —— MoE F2 dispatch 全套（30B 模型，需要 80 GB GPU）：

```bash
docker exec llava_megatron_container_ax bash -lc \
  "cd /workspace/LLaVA-OneVision-2.5 && PYTHONPATH=transformers_impl:. \
   OV2_REAL_FIXTURE=1 OV2_REAL_FIXTURE_MOE=1 \
   pytest tests/merge_ov2/moe/test_validators.py -v"
```

#### 4. 显存 / 内存测量脚本

```bash
bash tests/perf/measure_rss.sh       # dense merge 期间宿主机 RSS 峰值
bash tests/perf/measure_gpu_mem.sh   # dense merge 期间 GPU 显存峰值
```

不带 `OV2_REAL_FIXTURE=1` 会自动 skip。

---

## Package layout

```
transformers_impl/
├── merge_ov2.py             # 9-line BC shim → cli.main(["merge","--variant","dense", ...])
├── merge_ov2_moe.py         # 9-line BC shim → cli.main(["merge","--variant","moe",   ...])
└── merge_ov2/
    ├── __main__.py          # `python -m merge_ov2` entry (PYTHONPATH=transformers_impl:.)
    ├── cli.py               # argparse + 3 subcommands (merge / validate / dry-run)
    ├── loader.py            # strict in-place loader, dry-run report
    ├── io.py                # streaming safetensors iterator
    ├── remap.py             # ViT/Adapter/LLM key remap rules
    ├── save.py              # variant-aware save (copies modeling files)
    ├── utils.py             # log_stage timer, shared helpers
    ├── variants/
    │   ├── dense.py         # DenseVariant.build_empty(target_dtype, ...)
    │   └── moe.py           # MoeVariant.build_empty(target_dtype, ...)
    └── validators/
        ├── vit_blockorder.py    # whole-ViT forward parity
        ├── vit_layerwise.py     # per-block cosine parity (default for MoE)
        ├── llm_parallel.py      # original + merged side-by-side (dense only)
        ├── llm_sequential.py    # original then merged (default for MoE)
        └── e2e.py               # full multimodal forward + decode
```
