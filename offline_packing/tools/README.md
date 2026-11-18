# offline_packing / tools

Utilities for validating and tuning the packing pipeline.

## Files

| File | Purpose |
|---|---|
| `compare_patch_aware_packing.sh` | Run `auto_pipe.sh` twice on same input (baseline vs `--patch-slack`) and print side-by-side ViT load stats. |
| `analyze_bins.py` | Standalone analyzer for a `bins_*.pkl`. Prints token & patch (raw / vision-token) stats. |

Also see (one level up):
- `../verify_token_consistency.py` — sanity check: HF processor vs recorded `token_info.txt` on N random packed bins.
- `../smoke_test/run_smoke_test.sh` — single-pipeline 10k smoke test.

## Patch-count semantics

`patch_count` written by `s2_compute_token_lengths.py` and stored in
`bins_*.pkl` records is **raw ViT patches** (before spatial merge):

- NPY mode:  `npy.shape[0]`
- HF mode :  `image_grid_thw.prod(dim=1).sum() * merge_size**2`

LLM vision tokens = `raw_patches / merge_size**2` (e.g. 9 for merge=3).
We use **raw** for ViT load balancing because ViT attention runs over raw
patches before the spatial-merge projector. If you want to cap by
vision-token count instead, divide `--max-patches-per-bin` by `merge**2`.

## v2 (real-data) smoke test summary

10k real samples, `LLaVA-OneVision-2-4B-p16m33` (patch16, merge3, factor=48), L=32000:

|                    | Baseline | `--patch-slack 0.15` |
|---|---|---|
| Bins              | 195     | 195    |
| Token util mean   | 98.58%  | 98.58% |
| Patches CV        | 0.496   | 0.294  (-40.7%) |
| Max patches/bin   | 1.9M    | 0.93M  (-51.3%) |
| p99/p50 patches   | 2.68×   | 1.00×  |
| `verify_token_consistency.py` | 16/16 diff=0 | 16/16 diff=0 |
