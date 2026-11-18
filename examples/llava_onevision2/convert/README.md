# LLaVA-OneVision-2.5 Checkpoint Conversion

Scripts under this directory convert LLaVA-OneVision-2.5 checkpoints between
HuggingFace and Megatron-Core layouts for the 2B / 4B / 8B / 30B-A3B variants.

See the project [README](../../../README.md) for general usage. This document
covers one subtle point that has caused size-mismatch crashes in the past:
**how the ViT `patch_embed` is stored in Megatron checkpoints, and how the
converter handles it.**

---

## `PATCH_EMBED_TYPE` and on-disk layout

The vision tower (`aiak_training_llm/models/llava_onevision2/onevision_encoder_model.py`)
selects one of three patch-embedding implementations at training time via the
environment variable `PATCH_EMBED_TYPE`:

| `PATCH_EMBED_TYPE` | Module | TP behavior | Saved per-rank shape (`vision_model.patch_embed.proj.weight`) |
|---|---|---|---|
| `TP_LINEAR` (default) | `ParallelPatchEmbed` (`ColumnParallelLinear`) | sharded along dim 0 | `[embed_dim / tp, C, H, W]` (e.g. `[128, 3, 14, 14]` for embed=1024, TP=8) |
| `LINEAR` | `TorchLinearPatchEmbed` (`nn.Linear`) | replicated | `[embed_dim, C, H, W]`, identical on every TP rank |
| `CONV2D` | `PatchEmbed` (`nn.Conv2d`) | replicated | `[embed_dim, C, H, W]`, identical on every TP rank |

The HuggingFace checkpoint always expects `visual.embeddings.patch_embedding.weight`
of shape `[embed_dim, C, H, W]` (a plain Conv2d weight), so the converter must
reconstruct that single full tensor from the per-rank shards.

> **Note.** All three implementations save the patch weight as a 4-D Conv2d-shaped
> tensor for backward compatibility, even though `TP_LINEAR` and `LINEAR` use a
> Linear projection internally. The shape (full vs partitioned) is what tells
> the two layouts apart, not the rank.

## How `vision_patch.py` handles the two layouts

`tools/convert_checkpoint/custom/llava_onevision2/vision_patch.py` auto-detects
which layout it is reading:

1. Collects the per-TP-rank `patch_embed.proj.weight` from PP stage 0.
2. If all shards are bit-identical (`torch.equal` across ranks) → treats them as
   **replicated** (`LINEAR` / `CONV2D`) and keeps a single copy.
3. Otherwise → treats them as **TP-sharded** (`TP_LINEAR`) and concatenates
   along dim 0.

You can confirm which path was taken from the conversion log:

```
> patch_embed shards are REPLICATED across TP=8 (shape [1024, 3, 14, 14]); using rank 0 copy
```

or

```
> patch_embed shards are TP-SHARDED across TP=8 (per-rank shape [128, 3, 14, 14]); concatenating along dim 0
```

## Symptom of the old (pre-fix) bug

Previous versions of the converter unconditionally concatenated TP shards.
Converting a checkpoint trained with `PATCH_EMBED_TYPE=CONV2D` (or `LINEAR`)
under TP=N produced a weight of shape `[N * embed_dim, C, H, W]`, which then
fails to load into the HuggingFace model with:

```
RuntimeError: Error(s) in loading state_dict for Conv2d:
    size mismatch for weight: copying a param with shape torch.Size([8192, 3, 14, 14])
    from checkpoint, the shape in current model is torch.Size([1024, 3, 14, 14]).
```

`8192 = 8 * 1024` is the smoking gun: 8 identical replicas got cat'd along dim 0
under TP=8. The current `vision_patch.py` detects this and emits the
`REPLICATED` log line instead.

## Recommendation

Prefer `PATCH_EMBED_TYPE=TP_LINEAR` (the default) for new training runs. Under
TP=N it stores only `1/N` of the patch projection per rank, saving memory
versus `CONV2D` / `LINEAR` which hold a full copy on every TP rank. The
converter handles all three, so existing `CONV2D` / `LINEAR` checkpoints will
still convert correctly without any flag.
