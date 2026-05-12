# Current offline packing smoke test

This example is aligned with the current top-level implementation in `offline_packing/`.

It uses the canonical pipeline:

1. `s1_split_json_to_samples.py`
2. `s2_compute_token_lengths.py`
3. `s3_bin_packing.py`
4. `s4_bins_to_webdataset.py`

The wrapper is `offline_packing/auto_pipe.sh`.

## Files

- `sample_text_only.jsonl` — minimal text-only SFT/BMR-style records. These use `images: []`, so no image assets are required.
- `images/` — intentionally empty placeholder directory. `auto_pipe.sh` requires `--image-root`, even for text-only records.
- `run_auto_pipe_smoke.sh` — runnable command using the current pipeline.

## Run

Set `MODEL_PATH` to a local Qwen-VL processor/model directory that works with `transformers.AutoProcessor.from_pretrained(..., trust_remote_code=True)`.

```bash
cd offline_packing/examples_offline_packing/current_pipeline_smoke_test

MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
bash run_auto_pipe_smoke.sh
```

Optional overrides:

```bash
MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
OUTPUT_BASE=/tmp/offline_packing_smoke \
MAX_TOKEN_LEN=4096 \
bash run_auto_pipe_smoke.sh
```

## Expected outputs

The default output directory is `./output_smoke/`.

```text
output_smoke/
  s1_split_json2samples/
  token_info.txt
  bins_4096len.pkl
  bins_4096len_packing.log
  webdataset/
    smoke-000000.tar
    .nv-meta/
      dataset.yaml
      sample_loader.py
      split.yaml
```

## Switching to image data

Use the same JSONL structure, but set `images` to paths relative to `IMAGE_ROOT`:

```json
{"id":"image-0001","images":["demo.jpg"],"messages":[{"role":"user","content":"<image>\nDescribe the image."},{"role":"assistant","content":"A small demo image."}],"timestamp_decimal":1}
```

Then run:

```bash
MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
IMAGE_ROOT=/path/to/images \
bash run_auto_pipe_smoke.sh
```
