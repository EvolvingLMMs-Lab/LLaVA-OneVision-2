# Offline Packing Pipeline

Sequence-packing toolkit for multimodal SFT/pretraining data. Packs many
image/video samples into fixed-length token bins (e.g. 8k / 64k / 128k),
emits an [Energon](https://github.com/NVIDIA/Megatron-Energon) WebDataset
(`*.tar` + `*.tar.idx` + `.nv-meta/`) ready for training.

> Loss-equivalent to unpacked training (block-diagonal attention), while
> raising token utilization and cutting the number of forward sequences.

## Pipeline stages

| Stage | Script | Role |
|-------|--------|------|
| s0 | `s0_resample_from_master.py` | (video only) dynamic per-sample frame resampling from a 1fps content-md5 master frame store |
| s1 | `s1_build_manifest.py` | build lightweight manifest; normalize conversation schema (`from/value` → `role/content`) |
| s2 | `s2_compute_token_lengths.py` | compute exact per-sample token length with the model processor |
| s3 | `s3_bin_packing.py` | BFD bin-packing into fixed-capacity bins |
| s3b | `s3b_topup_with_47m.py` | *(optional)* fill leftover bin space with single-image samples |
| s4 | `s4_bins_to_webdataset.py` | encode bins → WebDataset tars (image downscale, B-mode) |
| — | `energon prepare` | build `.tar.idx` + `.nv-meta/.info.yaml` (run **inside** the training container) |

## Two entry points

### 1. Single-image / text SFT — `auto_pipe.sh`  (no s0)
```bash
bash auto_pipe.sh \
    -i  <input.jsonl> \        # one sample per line: {"images":[...], "messages":[{role,content},...]}
    -r  <image_root> \         # prefix for relative image paths ("/" if absolute)
    -o  <output_dir> \
    -L  8000 \                 # bin capacity (tokens); use 1 sample/bin with -M 1
    -M  200 \                  # max samples per bin (set 1 to DISABLE packing)
    --shard-prefix pack
```
- Multi-image and text-only samples are supported.
- `<image>` placeholder count **must** equal the number of images per sample.

### 2. Video (dynamic-budget "scheme-D") — `run_pack_master_cap.sh`
```bash
CAP=<bin_len> bash run_pack_master_cap.sh \
    <input_master.jsonl> \     # {"video_source":[...], "master_dir":[...]}  from the 1fps master store
    <output_dir> \
    <config.yaml>              # e.g. dynamic_budget_default.yaml (64k)
```
Runs s0 → s1..s4 → `energon prepare`. Frames come from an already-extracted
1fps master store (content-md5 addressed), so videos are **not re-decoded**.

## Dynamic-budget config (`dynamic_budget_*.yaml`)

Per-video, a `(resolution, fps)` pair is sampled by weight (reproducible via
`seed`); frames = `round(fps * duration)`, capped by a **text-aware** visual
budget so `visual + text <= bin`:

```
visual_budget = min(budget_tokens, hard_total_tokens - text_est - topup_safety_reserve)
```

| key | meaning |
|-----|---------|
| `resolution.values/weights` | per-frame pixel budgets to sample from |
| `fps.values/weights` | frame-rate options (<= master fps) |
| `budget_tokens` | hard cap on a single video's visual tokens |
| `hard_total_tokens` | bin total length |
| `topup_safety_reserve` | tokens kept free (for single-image top-up); `0` = pure video |
| `min_frames` / `max_frames` / `temporal_grid` | frame-count bounds (even multiples) |
| `chars_per_token_est` | text-token estimate = `ceil(chars / this)` |

Provided presets:
- `dynamic_budget_default.yaml` — 64k bin, with single-image top-up
- `dynamic_budget_128k.yaml` — 128k bin (long video), with top-up
- `dynamic_budget_bin110k_video.yaml` — 110k bin, **pure video** (no top-up)

## Output layout
```
<output_dir>/webdataset/
    pack-000000.tar ...          # packed bins
    pack-000000.tar.idx ...      # per-tar sample offsets (energon)
    .nv-meta/{.info.yaml, split.yaml, dataset.yaml, sample_loader.py}
```

## Important notes
- **`.tar.idx` is required for training.** `auto_pipe.sh` does not emit it by
  itself — run `energon prepare` inside the training container afterwards:
  ```bash
  cd <output_dir>/webdataset
  energon prepare --num-workers 12 --split-parts 'train:pack-{000000..999999}.tar' .
  ```
- **`-M 1` disables packing** (one sample per bin) — use it when you want each
  sample kept intact (e.g. benchmark-anchored eval sets), still shuffled and
  written as a WebDataset.
- For large corpora, shard the input and run one shard per machine; each
  `<output_dir>` is an independent Energon dataset.
- Timestamp tokens: for video, s2 accounts for the per-frame `<t seconds>`
  markers that training injects, so packed bins do not overflow the capacity.

## Tokenizer / model
Token lengths and the image processor come from the OneVision-2 model
directory (`--model-path`, default in `auto_pipe.sh`). Point it at your local
model snapshot.
