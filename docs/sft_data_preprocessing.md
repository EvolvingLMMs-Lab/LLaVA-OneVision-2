# SFT Data Preprocessing

This guide describes how to convert SFT data into the conversation JSON/JSONL
format used by LLaVA-OneVision and how to export it either as a standard
WebDataset or through the current offline-packing pipeline.

## 1. Docker environment

Using the project Docker image is recommended for reproducible preprocessing.

```bash
git clone https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2.git
cd LLaVA-OneVision-2

docker build -t llava_megatron:25.04 .

docker run -it --gpus all \
  --ipc host --net host --privileged --cap-add IPC_LOCK \
  --ulimit memlock=-1 --ulimit stack=67108864 --rm \
  -v "$(pwd)":/workspace/LLaVA-OneVision-2 \
  -w /workspace/LLaVA-OneVision-2 \
  --name llava_megatron_container \
  llava_megatron:25.04 /bin/bash
```

## 2. Download source data

Download LLaVA-NeXT-780k from
[HF / LLaVA-NeXT-780k](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data).

The example below assumes the parquet files are under:

```text
LLaVA-NeXT-Data/data/
```

## 3. Convert raw parquet files to LLaVA-OneVision JSON

The preprocessing scripts expect each sample to use this structure:

```json
{
  "id": "000000033471",
  "messages": [
    {
      "role": "user",
      "content": "<image>\nWhat are the colors of the bus in the image?"
    },
    {
      "role": "assistant",
      "content": "The bus in the image is white and red."
    }
  ],
  "images": ["000000033471.jpg"]
}
```

Use the following script to convert parquet rows into a single JSON file and an
image directory:

```python
import json
import os
from io import BytesIO

import pandas as pd
from PIL import Image
from tqdm import tqdm


PARQUET_DIR = "LLaVA-NeXT-Data/data"
OUTPUT_IMAGE_DIR = "images"
OUTPUT_JSON_FILE = "mllm_mix.json"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
merged_data = []

parquet_files = sorted(filename for filename in os.listdir(PARQUET_DIR) if filename.endswith(".parquet"))

for filename in tqdm(parquet_files):
    parquet_path = os.path.join(PARQUET_DIR, filename)
    df = pd.read_parquet(parquet_path, columns=["id", "conversations", "image"])

    for _, row in df.iterrows():
        messages = [
            {
                "content": message["value"],
                "role": "user" if message["from"] == "human" else "assistant",
            }
            for message in row["conversations"].tolist()
        ]

        sample = {
            "id": row["id"],
            "messages": messages,
            "images": [],
        }

        if row["image"] is not None:
            image = Image.open(BytesIO(row["image"]["bytes"]))
            extension = "jpg" if image.format in ["JPEG", "JPG"] else "png"
            image_name = f"{row['id']}.{extension}"
            image_path = os.path.join(OUTPUT_IMAGE_DIR, image_name)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
            sample["images"] = [image_name]

        merged_data.append(sample)

with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)
```

For JSONL workflows, write one JSON object per line instead of a single JSON
array. The current offline-packing pipeline accepts JSON or JSONL inputs.

## 4. Recommended path: offline packing

For SFT training at scale, use the maintained offline-packing pipeline. It
splits samples, computes token lengths, packs samples by token budget, and
converts packed bins directly to WebDataset shards.

```bash
bash offline_packing/auto_pipe.sh \
  -i /path/to/mllm_mix.json \
  -r /path/to/images \
  -m /path/to/Qwen2.5-VL-3B-Instruct \
  -o /path/to/output_base \
  -L 64000
```

Canonical stages:

1. `offline_packing/s1_split_json_to_samples.py`
2. `offline_packing/s2_compute_token_lengths.py`
3. `offline_packing/s3_bin_packing.py`
4. `offline_packing/s4_bins_to_webdataset.py`

A minimal runnable example is available at:

```text
offline_packing/examples_offline_packing/current_pipeline_smoke_test/
```

The only retained dataset-specific legacy reference is:

```text
offline_packing/examples_offline_packing/bmr_packing/
```

Use it only when you need to inspect the older multi-turn / mixed SFT BMR
formatting path.

## 5. Alternative path: standard WebDataset conversion

If you only need a standard, non-packed WebDataset, use the general conversion
script:

```bash
python tools/data_preprocess/convert_to_webdataset.py \
  --output_dir wds \
  --json_file mllm_mix.json \
  --image_dir images \
  --media image \
  --maxcount 10000
```

Key parameters:

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `--output_dir` | `str` | Yes | Directory for generated WebDataset shards. |
| `--json_file` | `str` | Yes | JSON file containing dataset metadata. |
| `--image_dir` | `str` | No | Image directory. Required when `--media` is `image` or `mix`. |
| `--video_dir` | `str` | No | Video directory. Required when `--media` is `video` or `mix`. |
| `--media` | `str` | No | Media type: `image`, `video`, or `mix`. Defaults to `mix`. |
| `--maxcount` | `int` | No | Maximum samples per WebDataset shard. Defaults to `10000`. |
| `--maxsize` | `int` | No | Maximum byte size per shard. Defaults to 3 GB. |
| `--columns_messages` | `str` | No | JSON key containing conversation messages. Defaults to `messages`. |

## 6. Verify conversion

Use `energon preview` to verify that the generated WebDataset can be read.

```bash
energon preview wds
```

If images and conversations are displayed correctly, the conversion succeeded.

<img src="../asset/wds_verification.png" style="max-width: 90%; height: auto;">
