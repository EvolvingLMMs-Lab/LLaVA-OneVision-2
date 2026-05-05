---
name: offline-packing-env-vars
description: Bilingual guide for the OFFLINE_PACKING_BMR and OFFLINE_PACKED_DATA environment variables that control LLaVA-OneVision2 training-side packing — what each gate does, why both must be enabled together, MBS=1 requirement, and the dead OFFLINE_PACKING_VQA branch
compatibility: opencode
metadata:
  domain: training-pipeline
  framework: llava-onevision2
  repo: llava-onevision2
---

## Purpose / 用途

Use this skill when you set up or debug **training-side** sample packing for LLaVA-OneVision2 — i.e. when you need to decide which env vars to export in a training shell script (Stage-1 / Stage-1.5 / Stage-2) and want to understand why both `OFFLINE_PACKING_BMR` and `OFFLINE_PACKED_DATA` must be `1` to actually get padding-free attention.

在配置或调试 LLaVA-OneVision2 **训练侧**的样本 packing 时使用——比如要决定在训练 shell 脚本（Stage-1 / Stage-1.5 / Stage-2）中导出哪些环境变量，以及为什么必须 `OFFLINE_PACKING_BMR=1` 和 `OFFLINE_PACKED_DATA=1` 同时打开才能真正获得 padding-free 的 attention。

This skill is specifically for:

- Choosing the correct env var combination in training scripts
- Diagnosing cross-sample attention leakage in packed runs
- Understanding why `cu_lengths` is a dummy `[[0]]` in some runs and a real `[B, P+1]` tensor in others
- Avoiding the well-known `OFFLINE_PACKING_VQA` red herring (it is dead code)

Companion skill: `cu-lengths-attention-flow` covers the **consumer** side (how `cu_lengths` is fed into ViT/LLM attention). This skill covers the **producer + gate** side.

姊妹 skill：`cu-lengths-attention-flow` 讲**消费端**（`cu_lengths` 如何送入 ViT/LLM attention）。本 skill 讲**生产端 + 开关**。

---

## TL;DR / 一句话总结

For packed training to work end-to-end, **both** env vars must be `1`:

```bash
export OFFLINE_PACKING_BMR='1'   # data-layer gate: build real cu_lengths
export OFFLINE_PACKED_DATA='1'   # batch-layer gate: forward real cu_lengths to model
```

Setting only one is a silent bug. `OFFLINE_PACKING_VQA` is **dead code**; do not rely on it.

---

## The Three Env Vars / 三个环境变量真相表

| Env var | Status | Default | Read at | Effect |
|---|---|---|---|---|
| `OFFLINE_PACKING_BMR` | **ALIVE** | `0` | `aiak_training_llm/data/multimodal/task_encoder.py:194` | Inside `PackedCaptioningSample` handling, unroll each packed entry into a `MultiMixQASample` (BMR-style, with full prompt/caption messages). When `0`, falls through to the legacy `CaptioningSample` branch which loses the multi-turn structure. |
| `OFFLINE_PACKED_DATA` | **ALIVE** | `0` | `aiak_training_llm/data/multimodal/task_encoder.py:363` | Inside `batch()`, replace dummy `cu_lengths = [[0]]` with the real per-sample `s.cu_lengths` stacked across the batch. Without this, the consumer side cannot construct `PackedSeqParams`. |
| `OFFLINE_PACKING_VQA` | **DEAD** | n/a | nowhere in `aiak_training_llm/` | Mentioned in README + several legacy shells under `examples/llava_onevision1_5/` and `examples/llava_onevision2/quick_start_video_2b/`, but **no source file reads it**. Setting it has zero runtime effect. Treat as documentation noise. |

> 💡 The `OFFLINE_PACKING_VQA` red herring is the #1 source of confusion. Newcomers see it in shell scripts and assume it controls VQA packing. It does not. There is no third packing branch in `task_encoder.py` — only the BMR branch and the legacy captioning fallback.

> 💡 `OFFLINE_PACKING_VQA` 这个红鲱鱼是头号困惑源。新人在 shell 脚本里看到它，以为它控制 VQA packing。**并不**。`task_encoder.py` 里没有第三个 packing 分支——只有 BMR 分支和老的 captioning fallback。

---

## Two-Stage Gate Architecture / 两段式 Gate 架构

Packing in this codebase is split into **two orthogonal gates** that must both fire. Understanding this is the whole point of the skill.

本仓库的 packing 拆成**两个正交的 gate**，必须都触发。理解这一点就是本 skill 的核心。

### Gate 1 — Data Layer (`OFFLINE_PACKING_BMR`)

**Where**: `aiak_training_llm/data/multimodal/task_encoder.py`, inside the `PackedCaptioningSample` branch of the encoder dispatch (`encode_sample` ~line 186).

**What it does**:
- For each entry inside the packed sample (`for idx in range(n_orig_sample):`), if `OFFLINE_PACKING_BMR == 1`, it builds a `MultiMixQASample` carrying the full chat-format messages (`{role: user, content: prompt}, {role: assistant, content: caption}`) and routes it through `encode_multi_mix_qa()`.
- If `OFFLINE_PACKING_BMR != 1`, it falls back to a plain `CaptioningSample` and `encode_captioning()` — losing the multi-turn / multi-image structure required for SFT.
- After the per-entry loop, **regardless of the BMR flag**, it calls `self.pack_selected_samples(l_Qwen2VLImageTaskSample)` (line 277), which constructs the per-sub-sample cumulative lengths `cu_lengths = [0, len₁, len₁+len₂, ...]` and attaches them to the resulting `ImageTaskSamplePacked` (line 473).

**Net effect**: enables the correct per-sub-sample encoding **and** produces real `s.cu_lengths` on each sample.

**作用**：启用正确的逐子样本编码，**并**在每个样本上产出真正的 `s.cu_lengths`。

> ⚠️ Even with BMR off, `pack_selected_samples` still attaches a `cu_lengths` tensor to the sample. But the sub-samples were encoded via the wrong path (legacy captioning), so the resulting boundaries don't match what the LLM actually sees. **BMR off + PACKED_DATA on is a hidden corruption**, not just a missing-feature.

> ⚠️ 即使 BMR 关掉，`pack_selected_samples` 仍然会给样本挂上 `cu_lengths` 张量。但子样本走的是错误的编码路径（老 captioning），结果 boundary 和 LLM 实际看到的 token 序列对不上。**BMR 关 + PACKED_DATA 开是隐性数据损坏**，不只是缺特性。

### Gate 2 — Batch Layer (`OFFLINE_PACKED_DATA`)

**Where**: `aiak_training_llm/data/multimodal/task_encoder.py:359-365`, inside `batch()` (the collate function).

**What it does**:

```python
# Cumulative sample lengths are needed for packing, otherwise use dummy values.
cu_lengths = torch.tensor([[0]], dtype=torch.int32)
max_lengths = torch.tensor([[0]], dtype=torch.int32)

if self.is_packing_enabled or int(os.environ.get("OFFLINE_PACKED_DATA", 0)) == 1:
    cu_lengths = torch.stack([s.cu_lengths for s in samples])
    max_lengths = torch.tensor([s.max_length for s in samples], dtype=torch.int32)
```

- Default: emit a dummy `cu_lengths` of shape `[1, 1]` containing only `[[0]]`.
- When `OFFLINE_PACKED_DATA == 1` (or the energon online-packing flag is set): stack the real per-sample `cu_lengths` produced by Gate 1 into shape `[B, P+1]`.

**Net effect**: decides whether the consumer (model forward) sees real packing offsets or a dummy that says "no packing".

**作用**：决定消费端（模型 forward）看到的是真实的 packing 偏移，还是一个表示"没有 packing"的 dummy。

### Why both gates must fire / 为什么必须两个都开

The consumer side at `aiak_training_llm/train/pretrain/pretrain_llava_onevision2.py:153-168`:

```python
packed_seq_params = None
...
if cu_lengths.shape == torch.Size([1, 1]):
    pass                        # treat as not packed
else:
    assert cu_lengths.shape[0] == 1, "micro-batch-size must be 1 for packing"
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_lengths[0],
        cu_seqlens_kv=cu_lengths[0],
        ...
    )
```

So:

| BMR | PACKED_DATA | Result |
|---|---|---|
| 0 | 0 | No packing. Each sample treated independently. Slow but correct (if data is unpacked). |
| 1 | 0 | **SILENT BUG.** Data is encoded as packed sub-samples (BMR), `cu_lengths` is built, but `batch()` discards it as dummy `[[0]]`. Consumer sees `shape == [1,1]` → `packed_seq_params = None` → flash-attn applies a single causal mask across the entire packed sequence → **cross-sub-sample attention leakage**. Loss looks fine; model silently learns wrong attention. |
| 0 | 1 | **HIDDEN CORRUPTION.** Sub-samples encoded via legacy path, boundaries in `cu_lengths` don't align with token sequence. Consumer applies varlen attention with wrong offsets. |
| 1 | 1 | **CORRECT.** BMR encodes properly, PACKED_DATA forwards the real offsets, consumer builds `PackedSeqParams`, flash-attn applies per-sub-sample causal mask via `cu_seqlens_q/kv`. |

> 🔥 The "BMR=1, PACKED_DATA=0" footgun is the most dangerous combination. Training does not crash. Loss curves look reasonable. But every sub-sample in a packed sequence can attend to every other sub-sample's prefix. Use this skill's TL;DR snippet to avoid it.

> 🔥 "BMR=1, PACKED_DATA=0" 这个组合最危险。训练不会挂，loss 曲线看着也正常。但 packed 序列里每个子样本都能 attend 到别的子样本的 prefix。用本 skill 顶部的 TL;DR 片段避开它。

---

## MBS=1 Hard Requirement / MBS=1 硬性要求

`pretrain_llava_onevision2.py:157`:

```python
assert cu_lengths.shape[0] == 1, "micro-batch-size must be 1 for packing"
```

When packing is on, `cu_lengths` has shape `[B, P+1]` where `B = micro_batch_size` and `P` = number of sub-samples in a packed sequence. The current `PackedSeqParams` construction only handles `B=1` (it indexes `cu_lengths[0]`). Therefore:

- `--micro-batch-size 1` is **mandatory** for any packed training run.
- Increase throughput via `--global-batch-size` (gradient accumulation), pipeline parallelism, or longer `--seq-length`, **not** via MBS.
- If you forget, the assert fires immediately on the first batch.

打开 packing 时，`cu_lengths` 形状是 `[B, P+1]`，B = micro batch size，P = 一个 packed 序列里的子样本数。当前 `PackedSeqParams` 构造只处理 `B=1`（取 `cu_lengths[0]`）。所以：

- 打包训练**必须** `--micro-batch-size 1`。
- 想提吞吐就调 `--global-batch-size`（梯度累积）、PP 并行、或更长的 `--seq-length`，**不要**调 MBS。
- 忘了的话第一个 batch 就 assert 挂掉。

---

## End-to-End Flow / 端到端流程图

```
┌─────────────────────────────────────────────────────────────────┐
│ Offline preprocessing (auto_pipe.sh, separate skill)            │
│   Produces WebDataset shards with PackedCaptioningSample format │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                    Energon dataloader yields PackedCaptioningSample
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ task_encoder.encode_sample()                                    │
│   if OFFLINE_PACKING_BMR == 1:                  ◄── GATE 1     │
│     for each sub-sample → MultiMixQASample → encode_multi_mix_qa │
│   else:                                                         │
│     for each sub-sample → CaptioningSample → encode_captioning  │
│   pack_selected_samples(l_samples)                              │
│     → ImageTaskSamplePacked with cu_lengths=[0,L1,L1+L2,...]   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ task_encoder.batch()                                            │
│   if is_packing_enabled or OFFLINE_PACKED_DATA==1:  ◄── GATE 2 │
│     cu_lengths = stack([s.cu_lengths for s in samples])         │
│   else:                                                         │
│     cu_lengths = [[0]]    # dummy, signals "not packed"         │
└──────────────────────────────┬──────────────────────────────────┘
                               │ batch dict broadcast via tensor_parallel
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ pretrain_llava_onevision2.get_batch_on_this_tp_rank()           │
│   if cu_lengths.shape == [1,1]: packed_seq_params = None        │
│   else:                                                         │
│     assert cu_lengths.shape[0] == 1   # MBS=1 required          │
│     packed_seq_params = PackedSeqParams(                        │
│       qkv_format="thd",                                         │
│       cu_seqlens_q=cu_lengths[0],                               │
│       cu_seqlens_kv=cu_lengths[0], ...)                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                Model forward → flash-attn varlen
                (see cu-lengths-attention-flow skill)
```

---

## Recipe: Correct Stage-N Script Snippet / 正确的训练脚本片段

```bash
# ───────────────────────────────────────────────────────────
# Packing env vars — both REQUIRED for padding-free training
# Set both to '1' when DATA_PATH points to offline-packed shards
#   (PackedCaptioningSample format, e.g. produced by auto_pipe.sh)
# Leave both as '0' (or unset) for unpacked datasets.
# Mixed states are silent bugs — see offline-packing-env-vars skill.
# ───────────────────────────────────────────────────────────
export OFFLINE_PACKING_BMR='1'
export OFFLINE_PACKED_DATA='1'

# Hard requirement when packing is on
MBS=1
# Throughput knobs: GBS via grad-accum, longer SEQ_LEN, more PP — not MBS
```

For an A/B control run that uses the **same packed dataset** but disables packing semantics (to measure the leakage cost), set both to `'0'`. Setting only `BMR=1` or only `PACKED_DATA=1` is **not a valid configuration** — it is a bug.

如果想做 A/B 对照，用**同一份 packed 数据**但关闭 packing 语义（为了量化 leakage 损失），**两个都设 `'0'`**。只开一个**不是合法配置**，是 bug。

---

## Concrete Stage-1 A/B Pair (this repo) / 本仓库的 Stage-1 A/B 对照

- `examples/llava_onevision2/quick_start_4b/stage_1_alignment_p16m3_packed.sh` — production: `BMR=1, PACKED_DATA=1`.
- `examples/llava_onevision2/quick_start_4b/stage_1_alignment_p16m3_packed_bmr_only.sh` — A/B control: `BMR=1, PACKED_DATA=0`. **Note**: this is the dangerous combo described above; it is named `bmr_only` deliberately to study the leakage effect, not as a recommended setting.

> If you copy `_bmr_only.sh` for a real production run, you will get cross-sub-sample attention leakage. Always confirm intent.

> 如果你把 `_bmr_only.sh` 拷去做正式训练，就会得到跨子样本 attention leakage。务必确认是有意为之。

---

## Diagnostics / 排查清单

If your packed training looks "off" (loss too low / too smooth / model overfits prefixes):

1. `grep -n 'OFFLINE_PACKING_BMR\|OFFLINE_PACKED_DATA' your_script.sh` — both should be `'1'`.
2. Add a one-shot print in `task_encoder.batch()` after line 365: `print('cu_lengths.shape:', cu_lengths.shape)`. Expect `[1, P+1]` with `P >= 2`. If you see `[1, 1]`, Gate 2 is closed.
3. Add a print in `pretrain_llava_onevision2.py` after line 168: `print('packed_seq_params:', packed_seq_params)`. Should be a real `PackedSeqParams`, not `None`.
4. Confirm `MBS=1` in the shell (`--micro-batch-size 1`). Otherwise the assert at line 157 fires and you wouldn't be reading this.
5. Confirm dataset is actually packed: `cat $DATA_PATH/.../webdataset/.nv-meta/.info.yaml` — look for shard structure produced by `auto_pipe.sh` (PackedCaptioningSample).
6. **Do not** add `OFFLINE_PACKING_VQA=1` thinking it helps. It does nothing in this codebase.

---

## Cross-References / 交叉引用

- **Producer pipeline** (how the packed shards are built): `distributed-offline-packing` skill.
- **Consumer attention semantics** (how `cu_lengths` is interpreted by ViT and LLM): `cu-lengths-attention-flow` skill.
- **Dataloader length-balancing** across ranks: `length-pool-sort-dataset` skill.

## Source File Index / 源文件索引

| File | Lines | What |
|---|---|---|
| `aiak_training_llm/data/multimodal/task_encoder.py` | 186-279 | `PackedCaptioningSample` branch + Gate 1 (`OFFLINE_PACKING_BMR`) |
| `aiak_training_llm/data/multimodal/task_encoder.py` | 359-365 | `batch()` Gate 2 (`OFFLINE_PACKED_DATA`) |
| `aiak_training_llm/data/multimodal/task_encoder.py` | 401-477 | `pack_selected_samples` — builds real `cu_lengths` |
| `aiak_training_llm/train/pretrain/pretrain_llava_onevision2.py` | 145-168 | Consumer: `cu_lengths.shape` check + `PackedSeqParams` construction + MBS=1 assert |
| `aiak_training_llm/train/pretrain/pretrain_llava_onevision2.py` | 171-207 | SP padding for `packed_seq_params` (TP/SP-only path) |
