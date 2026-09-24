# Dynamic packing pipeline (scheme-D dynamic budget)

> Offline packing for a long-video master frame store: allocate a per-sample visual-token budget dynamically so each video fills a bin as tightly as possible, precisely fill the leftover slack with 47m single images, and finally emit an energon-format webdataset.

- **Driver script**: `run_pack_master_47m.sh`
- **Configs**: `dynamic_budget_default.yaml` (64k) / `dynamic_budget_128k.yaml` (long-video 128k)
- **Core script**: `s0_resample_from_master.py`
- **Directory**: `offline_packing/` (this directory)

---

## 0. Core idea

Unlike static packing ("fixed resolution / fixed frame count per sample"), scheme-D **draws one (resolution, fps) pair per video using a reproducible random seed**, then decides how many frames to sample and how large each frame is, so a single video's visual tokens approach the bin capacity. It is also **text-aware**: it reserves tokens for the caption and fills the remaining space with 47m single images.

Key quantities:
- `CAP`  : bin sequence length (default `64000`; `128000` for long videos)
- `MAXP` : bin patch-count cap = `608822 * CAP / 64000`

---

## 1. Prerequisite: decode-once 1fps master frame store

Every video is **decoded only once**, sampled at `master_fps=1` for all frames, and stored in a content-md5-addressed frame store
(`<root>/fps1/<md5[:2]>/<md5[2:4]>/<md5>/frame_XXXXXX.jpg + master_meta.json`).
`master_meta.json` holds `native_fps` (the real video fps), `num_frames` (= duration in seconds, since 1fps), etc.
Dynamic packing "resamples" on top of this store; it never re-decodes the video.

---

## 2. Stage chain (7 steps, run in order)

### s0 — dynamic resample  ★core (`s0_resample_from_master.py`)
For each video:
1. **Weighted random sampling** (RNG = f(seed, sample_id) → random but reproducible):
   - resolution ∈ {150k, 200k, 250k, 300k} px, weights {.30, .30, .25, .15}
   - fps ∈ {1.0, 0.7, 0.5, 0.33}, weights {.40, .30, .20, .10} (cap 1.0 = master fps)
2. **Text budget**: `text_tok ≈ ceil(caption_chars / chars_per_token_est=3.0)` (conservative, never underestimates)
3. **Visual budget (text-aware)**:
   ```
   visual_budget = min( budget_tokens=55000,
                        hard_total=64000 − text_tok − reserve=2000 )
   ```
   → a long caption automatically shrinks the visual budget, so visual+text never busts 64k (otherwise s2 would drop the sample)
4. **Frame count**: `frames = round(fps × duration)`, capped by visual_budget
   (tokens/frame = resolution / (patch_size² × factor²) = resolution / 2304),
   clamped to `[min_frames=200, max_frames=960]`, rounded to an even multiple of `temporal_grid=2`
5. **Sample N frames evenly** from the master store; expand the `<video>` in the first user turn into `<image>×N`;
   write the real `fps` (native_fps, so training can inject timestamps), `patch_positions`, etc.
6. **Per-frame target size**: `resize_hw` = one `[h, w]` per frame (smart_resize of the master frame size to the
   sampled resolution, aligned to patch_size × factor = 48). `patch_positions` are built for exactly this size.
   Master frames stay at master resolution on disk; s4 downscales each frame to its `resize_hw` on read
   (`ImagePolicy.apply_to_bytes_exact`), independent of `--no-preprocess`.
- Output: `s0_expanded.jsonl` (frame count / resolution fixed per video)

### s1 — build manifest (`s1_build_manifest.py`)
Convert to a lightweight manifest (paths + conversation + patches) and normalize the conversation schema (`from/value → role/content`).
- Output: `manifest.jsonl`

### s2 — compute token lengths (`s2_compute_token_lengths.py`)
Use the OV2 processor to compute the exact token count per sample (image + text).
- Output: `token_info.txt` (`name:tokens:patches`)

### s3 — bin packing (`s3_bin_packing.py`, BFD, 1 video/bin)
`--max-samples 1 --algorithm bfd --capacity $CAP`:
each bin holds only **1 video** (a single video is already ~55k, close to 64k), BFD-packed and **deliberately leaving slack** (~9k) for the next step.
- Output: `bins_64klen.pkl`

### s3b — 47m topup  ★ (`s3b_topup_with_47m.py`)
Randomly fill each bin's leftover space with **47m single images** up to `CAP`:
- pool: `<47m_pool>/47m_part*/token_info.txt` (single-image pool, given by `--pool-tokeninfo-glob`)
- `--pool-shard-index/--pool-shard-count`: **disjoint sharding** — multi-host / multi-batch packing takes non-overlapping subsets, preventing 47m duplication
- `--pool-blocklist`: cross-batch dedup; `--image-only`; `--max-patches $MAXP`; `--max-samples 200`
- Output: `bins_topup.pkl` (video + 47m mix)

### combine — merge manifests (`tools/build_combined_manifest.py`)
Video manifest + selected 47m manifest → one combined manifest for s4 to read bytes from.
- Output: `manifest_combined.jsonl`

### s4 — build webdataset  ★ (in container, `s4_bins_to_webdataset.py`)
`docker run llava_megatron:26.05` runs **inside the container**:
- read frame/image bytes per bin → B-mode downscale (`--longest-edge 3072`) → write tar
- **in-container energon** generates `.tar.idx` + `.info.yaml` (required for energon training loading)
- docker mounts must include `<data_root>` (frame store + single-image source disk; mount with docker -v as needed)
- Output: `webdataset/` (`*.tar` + `*.tar.idx` + `.nv-meta/{.info.yaml, split.yaml, dataset.yaml, sample_loader.py}`)

---

## 3. Config reference (`dynamic_budget_default.yaml`, 64k)

```yaml
mode: dynamic
resolution:                 # min_pixels = max_pixels, sampled by weight
  values:  [150000, 200000, 250000, 300000]
  weights: [0.30, 0.30, 0.25, 0.15]
fps:                        # cap 1.0 (matches master 1fps)
  values:  [1.0, 0.7, 0.5, 0.33]
  weights: [0.40, 0.30, 0.20, 0.10]
budget_tokens: 55000        # per-video visual-token cap (leaves slack for 47m + text)
min_frames: 200             # frame-count floor
max_frames: 960             # frame-count hard cap (budget usually caps first)
patch_size: 16
factor_multiplier: 3        # tok/frame = resolution / (patch² × factor²) = resolution / 2304
min_pixels_floor: 3136      # 56×56, smart_resize floor
temporal_grid: 2            # frame count rounded to an even multiple of temporal_grid
seed: 20260723              # base seed; per-sample RNG = f(seed, sample_id) → random & reproducible
hard_total_tokens: 64000    # bin hard cap
topup_safety_reserve: 2000  # reserved for 47m topup + safety margin
chars_per_token_est: 3.0    # text tokens ≈ ceil(chars / 3.0), conservative
```
> For 128k long video use `dynamic_budget_128k.yaml`: `budget_tokens=110000`, `hard_total=128000`, `max_frames=1536`.

---

## 4. How to call

```bash
bash run_pack_master_47m.sh <input_master_jsonl> <output_base> \
     dynamic_budget_default.yaml <shard_idx> <shard_cnt>
```
Environment knobs:
- `CAP`         : bin length (default 64000; 128000 for long videos)
- `S2_WORKERS`  / `S4_WORKERS` / `S4_CPUS` : per-stage parallelism (lower when borrowing training hosts)
- `BLOCKLIST`   : 47m cross-batch dedup list
- `<shard_idx> <shard_cnt>` : 47m pool disjoint shard (differs per host for multi-host / multi-batch packing)

---

## 5. Difference vs the single-image chain (auto_pipe.sh)

| | dynamic chain (scheme-D) | auto_pipe (single-image) |
|---|---|---|
| s0 resample | ✅ dynamic frames/resolution | ❌ none (not needed for single image) |
| bin packing | 1 video/bin + 47m fill | multiple samples/bin |
| s4 | **in container**, emits `.tar.idx` | bare run, **no idx** (needs a later `energon prepare`) |
| use case | long-video master | single-image SFT / caption |

---

## 6. One-line flow

```
decode-once 1fps master
  → s0  dynamic resample (text-aware frame count/resolution, reproducible random)
  → s1  manifest
  → s2  token lengths
  → s3  BFD one video per bin (leave slack)
  → s3b 47m disjoint fill
  → combine merge manifests
  → s4  in-container encode + energon index
  → 64k webdataset (tar + idx + nv-meta)
```
