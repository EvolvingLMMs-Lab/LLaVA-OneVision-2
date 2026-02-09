<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/llava_onevision_black.png">
    <source media="(prefers-color-scheme: light)" srcset="asset/llava_onevision_white.png">
    <img alt="LLaVA-OneVision-1.5" src="output/llava_onevision_white.png" width="600" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
  <strong>Fully Open Framework for Democratized Multimodal Training</strong>
</p>



<div align="center">

🤗 **[Models and Datasets](https://huggingface.co/collections/lmms-lab/llava-onevision-15-68d385fe73b50bd22de23713)** |
🖥️ **[Demo](https://huggingface.co/spaces/lmms-lab/LLaVA-OneVision-1.5)** |
📄 **[Technical Report](https://arxiv.org/abs/2509.23661)** |
📰 **[Zhihu](https://www.zhihu.com/question/1959577143697707446)** |
📕 **[Xiaohongshu](http://xhslink.com/o/4nXL6EXDTqv)**

</div>

---

<p align="center">
  <!-- Mid-Training Dataset Downloads -->
  <a href="https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M">
    <img alt="HF Mid-Training Dataset Downloads" src="https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M&amp;query=downloads&amp;label=Mid%20Training%20DATA%20Downloads&amp;color=green&amp;logo=huggingface&amp">
  </a>
  <!-- Instruct Dataset Downloads -->
  <a href="https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data">
    <img alt="HF Instruct Dataset Downloads" src="https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data&amp;query=downloads&amp;label=Instruct%20DATA%20Downloads&amp;color=blue&amp;logo=huggingface&amp">
  </a>
  <!-- Model Downloads -->
  <a href="https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct">
    <img alt="HF Model Downloads" src="https://img.shields.io/badge/dynamic/json?url=https://huggingface.co/api/models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct&amp;query=downloads&amp;label=OV-1.5-8B-Instruct%20Downloads&amp;color=yellow&amp;logo=huggingface&amp">
  </a>
  <!-- Training Cost -->
  <img alt="Training Cost" src="https://img.shields.io/badge/Full%20Train%20Cost-~$16K-success">
  <!-- Paper Citations -->
  <a href="https://scholar.google.com/scholar_lookup?arxiv_id=2509.23661">
    <img alt="Citations" src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2FARXIV%3A2509.23661%3Ffields%3DcitationCount&amp;query=citationCount&amp;label=Citations&amp;color=orange&amp;logo=googlescholar&amp">
  </a>
  <!-- License -->
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-Apache--2.0-blue.svg?logo=apache&amp">
  </a>
  <!-- PRs Welcome -->
  <a href="https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5/pulls">
    <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?logo=github&amp">
  </a>
  <!-- Commit Activity -->
  <a href="https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5/commits">
    <img alt="Commit Activity" src="https://img.shields.io/github/commit-activity/m/EvolvingLMMs-Lab/LLaVA-OneVision-1.5?logo=github&amp">
  </a>
  <!-- Contributors -->
  <a href="https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5/graphs/contributors">
    <img alt="Contributors" src="https://img.shields.io/github/contributors/EvolvingLMMs-Lab/LLaVA-OneVision-1.5?logo=github&amp">
  </a>
  <!-- Megatron-LM Optimization -->
  <a href="https://github.com/NVIDIA/Megatron-LM">
    <img src="https://img.shields.io/badge/Megatron--LM-mcore%20optimized-1560b9?logo=nvidia&amp" alt="Megatron-LM mcore optimized">
  </a>
  <!-- ModelScope Collection -->
  <a href="https://www.modelscope.cn/collections/LLaVA-OneVision-15-ff6ede3d20a643" target="_blank">
    <img alt="ModelScope Collection" src="https://img.shields.io/badge/ModelScope-Collection-orange?logo=modelscope">
  </a>
</p>

---


## NEWS
- 2025-12-11: Released [RL recipe for LLaVA-OneVision-1.5](https://mvp-ai-lab.github.io/LLaVA-OneVision-1.5-RL/) with [code](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5-RL), [data](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-RL-Data), and [model](https://huggingface.co/mvp-lab/LLaVA-OneVision-1.5-8B-RL).
- 2025-09-30: Released the [Offline Data Packing Guide](examples_offline_packing).
- 2025-09-30: Released the LLaVA-OneVision-1.5 [Technical Report](https://arxiv.org/abs/2509.23661).


## Contents
<!-- TOC (no expansion for Quick Start Guide / Fully Reproducing Guide) -->
- [Introduction](#introduction)
- [Models](#models)
- [Datasets](#datasets)
- [Results](#evaluation-results)
- [Quick Start with Hugging Face](#quick-start-with-huggingface)
- [Evaluation](#evaluation)
- [Quick Start For Training](#quick-start-guide)
- [Fully Reproducing Guide](#fully-reproducing-guide)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


<!-- ## Introduction
**LLaVA-OneVision-1.5** introduces a family of fully open-source large multimodal models (LMMs) that operate on **native-resolution images**, achieve **state-of-the-art** performance, and require comparatively **lower training costs**.

#### **Superior Performance**
  - The model leads on multiple multimodal benchmarks and generally surpasses Qwen2.5-VL.
  - Training on native-resolution images significantly improves its visual understanding.

#### **High-Quality Data at Scale**
  - The pretraining corpus comprises large-scale, concept-balanced, diverse, and high-quality captions curated with strict filtering and quality control.
  - The instruction-tuning dataset is comprehensive and covers a wide range of tasks.

#### **Ultra-Efficient Training Framework**
  - The end-to-end training cost is about $16,000 on A100 GPUs at roughly $0.60 per GPU-hour.
  - The system is built on Megatron-LM with support for MoE, FP8, and long-sequence parallelism, and the codebase is optimized for cost-effective scaling.

#### **Fully Open Framework**
  - The project releases high-quality pretraining and SFT datasets along with the complete training framework, configurations, and recipes.
  - It also provides detailed training logs and metrics to enable reproducibility and community adoption.


## Models

| Model                    | HF Link                                                                                      | Training Log |
|--------------------------|--------------------------------------------------------------------------------------------------------|-------------|
| LLaVA-OneVision-1.5-4B-Instruct | [🤗 HF / 4B-Instruct](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct)                | [📈 TensorBoard](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct/tensorboard) |
| LLaVA-OneVision-1.5-8B-Instruct | [🤗 HF / 8B-Instruct](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct)                | [📈 TensorBoard](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct/tensorboard) |
| LLaVA-OneVision-1.5-4B-Base     | [🤗 HF / 4B-Base](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Base)                        | [📈 TensorBoard](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct/tensorboard) |
| LLaVA-OneVision-1.5-8B-Base     | [🤗 HF / 8B-Base](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Base)                        | [📈 TensorBoard](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct/tensorboard) |
## Datasets

![Dataset Visualization](asset/dataset.jpg)
<p align="left">
  <strong>(a)</strong> The vocabulary coverage proportion in the LLaVA-OneVision-1.5 Mid-Training dataset before and after concept balancing.
  <strong>(b)</strong> Distribution of data sources within the LLaVA-OneVision-1.5 Mid-Training dataset.
  <strong>(c)</strong> Distribution of data sources within the LLaVA-OneVision-1.5 Instruct dataset.
</p>

| Description        | Link                                                                                                   | Status      |
|--------------------|--------------------------------------------------------------------------------------------------------|-------------|
| LLaVA-OneVision-1.5-Mid-Training-85M   | [🤗HF / Mid-Training 85M](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M) | Available  |
| LLaVA-OneVision-1.5-Instruct           | [🤗HF / Instruct-Data](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)        | Available  |


## Evaluation Results


All evaluations were conducted using [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

![](asset/performance.png)


## Quick Start with HuggingFace

```python
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info
model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"

# default: Load the model on the available device(s)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# default processor
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

``` -->

## Evaluation
```
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git  

accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision1_5 \
    --model_args=pretrained=lmms-lab/LLaVA-OneVision-1.5-8B-Instruct,attn_implementation=flash_attention_2,max_pixels=3240000 \
    --tasks=mmmu_val,mmmu_pro_standard,mmbench_en_test,mmerealworld,mmerealworld_cn,ai2d,ai2d_no_mask,vstar_bench,chartqa,charxiv,docvqa_test,mathvista_testmini,mmstar,scienceqa \
    --batch_size=1
```


## Quick Start Guide

### 1.🐳 Docker (Recommended)

We strongly recommend using the docker environment for a seamless experience. The following instructions are tailored for the A100 80GB GPU environment.


```bash
# Clone repository
git clone https://github.com/anxiangsir/LLaVA-OneVision-2.git
cd LLaVA-OneVision-2

docker build -t llava_megatron:25.12 .

# Run container with -w to set working directory directly to the mounted volume
docker run -it --gpus all \
    --ipc host --net host --privileged --cap-add IPC_LOCK \
    --ulimit memlock=-1 --ulimit stack=67108864 --rm \
    -v $(pwd):/workspace/LLaVA-OneVision-2 \
    -w /workspace/LLaVA-OneVision-2 \
    --name "llava_megatron_container" \
    llava_megatron:25.12 /bin/bash
```

### 2. Checkpoint and Format Conversion

You have two options to get started with LLaVA-OneVision-1.5-stage-0:

<!-- #### Option 1: Download pre-trained model from Hugging Face
Download our `LLaVA-OneVision-1.5-4B-stage0` model directly from [Hugging Face](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-stage0). -->

#### Option 2: Merge initial weights yourself
Alternatively, you can merge the initial weights from the original ViT and LLM:
```bash
python ds/merge_model.py \
--vit_path DeepGlint-AI/rice-vit-large-patch14-560 \
--llm_path Qwen/Qwen3-4B-Instruct-2507 \
--output LLaVA-OneVision-1.5-4B-stage0
```
Note: When merging weights, the adapter component will be initialized with default values.

Convert the model from Hugging Face format to Megatron format:

```bash
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2 bash examples/llava_onevision1_5/convert/convert_4b_hf_to_mcore.sh \
LLaVA-OneVision-1.5-4B-stage0 \
LLaVA-OneVision-1.5-4B-stage0_mcore_tp1_pp1 \
1 1
```

### 3. Stage 1 Alignment-Training

Download LLaVA from [LLaVA-558K-Webdataset](https://huggingface.co/datasets/lmms-lab/LLaVA-558K-Webdataset).


```bash
# ============================================================
# Required environment variables:
#   AIAK_TRAINING_PATH  Root directory of the AIAK-Training-LLM project
#   DATA_PATH           Directory with WebDataset shards (.tar) for pretraining
#   TOKENIZER_PATH      Hugging Face tokenizer directory
#   CHECKPOINT_PATH     Megatron-formatted checkpoint directory (e.g., mcore TP1/PP1)
#   SAVE_CKPT_PATH      Output directory for saving training checkpoints
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2 \
DATA_PATH=LLaVA-558K-Webdataset \
TOKENIZER_PATH=LLaVA-OneVision-1.5-4B-stage0 \
CHECKPOINT_PATH=LLaVA-OneVision-1.5-4B-stage0_mcore_tp1_pp1 \
bash examples/llava_onevision1_5/quick_start/stage_1_alignment_llava_ov_4b.sh
```

### 4. Stage 1.5 Mid-Training 

Download our lightweight packed subset from [LLaVA-OneVision-1.5-Mid-Training-Quick-Start-3M-Webdataset](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-1.5-Mid-Training-Webdataset-Quick-Start-3M).

```bash
# ============================================================
# Convert model to release format
bash examples/llava_onevision1_5/convert/convert_4b_mcore_to_release.sh \
stage_1_alignment_llava_ov_4b/iter_0002500/ \
stage_1_alignment_llava_ov_4b_release 1 1
# ============================================================
# Launch
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2 \
DATA_PATH=LLaVA-OneVision-1.5-Mid-Training-Webdataset-Quick-Start-3M \
TOKENIZER_PATH=LLaVA-OneVision-1.5-4B-stage0 \
CHECKPOINT_PATH=stage_1_alignment_llava_ov_4b_release \
bash examples/llava_onevision1_5/quick_start/stage_1.5_mid_training_llava_ov_4b.sh
```


### 5. Stage 2 Instruct-Training

Download LLaVA-NeXT-780k-webdataset at [LLaVA-NeXT-780K Dataset](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-780k-webdataset).

```bash
# ============================================================
# Convert model to release format
bash examples/llava_onevision1_5/convert/convert_4b_mcore_to_release.sh \
stage_1.5_mid_training_llava_ov_4b/iter_0020000/ \
stage_1.5_mid_training_llava_ov_4b_release 1 1
# ============================================================
# # Launch
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2 \
DATA_PATH=LLaVA-NeXT-780k-Webdataset \
TOKENIZER_PATH=LLaVA-OneVision-1.5-4B-stage0 \
CHECKPOINT_PATH=stage_1.5_mid_training_llava_ov_4b_release \
bash examples/llava_onevision1_5/quick_start/stage_2_instruct_llava_ov_4b.sh
```


### 6. Convert mcore to Hugging Face
```bash
AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2 \
bash examples/llava_onevision1_5/convert/convert_4b_mcore_to_hf.sh \
stage_2_instruct_llava_ov_4b/iter_0003500 \
LLaVA-OneVision-1.5-4B-3M-Mid-Training-780K-Instruct \
1 1
# Copy non-model files (e.g., tokenizer config) to the new directory
find LLaVA-OneVision-1.5-4B-stage0/ -type f -not -iname '*safetensors*' -exec cp {}  LLaVA-OneVision-1.5-4B-3M-Mid-Training-780K-Instruct/ ';'
```

### 7. Evaluation
```bash
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch \
--num_processes=4 --main_process_port 12399 -m lmms_eval --model=llava_onevision1_5 --batch_size=1 --tasks=mme \
--model_args=pretrained=/workspace/LLaVA-OneVision-2/LLaVA-OneVision-1.5-4B-3M-Mid-Training-780K-Instruct,max_pixels=3240000
```

## Fully Reproducing Guide

> [!TIP]
> More detailed reproduction steps for the complete process will be provided after the dataset upload is completed.


### Mid-Training

To improve model training efficiency, we implement offline sample packing:

1. Download the [**Mid-Training-85M Dataset**](https://huggingface.co/datasets/lmms-lab/LLaVA-One-Vision-1.5-Mid-Training-85M)
2. Pack the data into WebDataset format, refer to [**Examples offlinepacking**](examples_offline_packing) and [**Offline Padding-Free Data Packing**](examples/llava_onevision1_5/sample_packing/README.md)


### Instruct
1. Download the [**LLaVA-OneVision-1.5-Instruct-Data**](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-1.5-Instruct-Data)
2. Convert the data into WebDataset format, refer to [**Conversion for Mixed Instruction Data**](docs/sft_data_preprocessing.md)

## Roadmap

Q4 2025 Key Deliverables:

1. **Ultra-efficient MoE Training**  
2. **Full Video Input LLM**  


## Contributors
Thanks so much to all of our amazing contributors!

<!-- readme: collaborators,contributors,jiankangdeng/- -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/fengshikun">
                    <img src="https://avatars.githubusercontent.com/u/2499990?v=4" width="80;" alt="fengshikun"/>
                    <br />
                    <sub><b>fengshikun</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/GeoffreyChen777">
                    <img src="https://avatars.githubusercontent.com/u/14183213?v=4" width="80;" alt="GeoffreyChen777"/>
                    <br />
                    <sub><b>GeoffreyChen777</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/fdcp">
                    <img src="https://avatars.githubusercontent.com/u/15667917?v=4" width="80;" alt="fdcp"/>
                    <br />
                    <sub><b>fdcp</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Luodian">
                    <img src="https://avatars.githubusercontent.com/u/15847405?v=4" width="80;" alt="Luodian"/>
                    <br />
                    <sub><b>Luodian</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/mathCrazyy">
                    <img src="https://avatars.githubusercontent.com/u/20607153?v=4" width="80;" alt="mathCrazyy"/>
                    <br />
                    <sub><b>mathCrazyy</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/anxiangsir">
                    <img src="https://avatars.githubusercontent.com/u/31175974?v=4" width="80;" alt="anxiangsir"/>
                    <br />
                    <sub><b>anxiangsir</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/didizhu-judy">
                    <img src="https://avatars.githubusercontent.com/u/34787894?v=4" width="80;" alt="didizhu-judy"/>
                    <br />
                    <sub><b>didizhu-judy</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/yiyexy">
                    <img src="https://avatars.githubusercontent.com/u/35927125?v=4" width="80;" alt="yiyexy"/>
                    <br />
                    <sub><b>yiyexy</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/YunyaoYan">
                    <img src="https://avatars.githubusercontent.com/u/109638667?v=4" width="80;" alt="YunyaoYan"/>
                    <br />
                    <sub><b>YunyaoYan</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/FeilongTangmonash">
                    <img src="https://avatars.githubusercontent.com/u/152372878?v=4" width="80;" alt="FeilongTangmonash"/>
                    <br />
                    <sub><b>FeilongTangmonash</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/chengzheng345">
                    <img src="https://avatars.githubusercontent.com/u/209475443?v=4" width="80;" alt="chengzheng345"/>
                    <br />
                    <sub><b>chengzheng345</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/wideyard">
                    <img src="https://avatars.githubusercontent.com/u/101321826?v=4" width="80;" alt="wideyard"/>
                    <br />
                    <sub><b>wideyard</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Lornatang">
                    <img src="https://avatars.githubusercontent.com/u/31124350?v=4" width="80;" alt="Lornatang"/>
                    <br />
                    <sub><b>Lornatang</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/killTheHostage">
                    <img src="https://avatars.githubusercontent.com/u/16442720?v=4" width="80;" alt="killTheHostage"/>
                    <br />
                    <sub><b>killTheHostage</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/wkzhang636">
                    <img src="https://avatars.githubusercontent.com/u/194186498?v=4" width="80;" alt="wkzhang636"/>
                    <br />
                    <sub><b>wkzhang636</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/yunglechao">
                    <img src="https://avatars.githubusercontent.com/u/7631185?v=4" width="80;" alt="yunglechao"/>
                    <br />
                    <sub><b>yunglechao</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/RobitYadda">
                    <img src="https://avatars.githubusercontent.com/u/6811311?v=4" width="80;" alt="RobitYadda"/>
                    <br />
                    <sub><b>RobitYadda</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: collaborators,contributors,jiankangdeng/- -end -->

## Citation

If you find *LLaVA-OneVision-1.5* useful in your research, please consider to cite the following related papers:

```
@inproceedings{LLaVA-OneVision-1.5,
  title={LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training},
  author={An, Xiang and Xie, Yin and Yang, Kaicheng and Zhang, Wenkang and Zhao, Xiuwei and Cheng, Zheng and Wang, Yirui and Xu, Songcen and Chen, Changrui and Wu, Chunsheng and Tan, Huajie and Li, Chunyuan and Yang, Jing and Yu, Jie and Wang, Xiyao and Qin, Bin and Wang, Yumeng and Yan, Zizhen and Feng, Ziyong and Liu, Ziwei and Li, Bo and Deng, Jiankang},
  booktitle={arXiv},  
  year={2025}
 }

@inproceedings{xie2025region,
  title={Region-based Cluster Discrimination for Visual Representation Learning},
  author={Xie, Yin and Yang, Kaicheng and An, Xiang and Wu, Kun and Zhao, Yongle and Deng, Weimo and Ran, Zimin and Wang, Yumeng and Feng, Ziyong and Miles, Roy and Elezi, Ismail and Deng, Jiankang},
  booktitle={ICCV},
  year={2025}
}

@article{lillava,
  title={LLaVA-OneVision: Easy Visual Task Transfer},
  author={Li, Bo and Zhang, Yuanhan and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Hao and Zhang, Kaichen and Zhang, Peiyuan and Li, Yanwei and Liu, Ziwei and Li, Chunyuan},
  journal={Transactions on Machine Learning Research}
  year={2024}
}
```

## Acknowledgement

We extend our sincere gratitude to **AIAK team of the** [**Baige AI computing platform**](https://cloud.baidu.com/product/aihc.html) **from Baidu AI Cloud** for providing the exceptional training framework. The outstanding capabilities of AIAK-Training-LLM and AIAK-Megatron have significantly accelerated our training process with remarkable efficiency. These cutting-edge frameworks have been instrumental in achieving our research goals. `To get full AIAK support, you can contact Baidu Cloud.` 

We acknowledge the support of [Synvo AI](https://synvo.ai/) for contributing to the partial data annotation in this work, and also thank the maintainers and contributors of the following open-source projects, whose work greatly inspired and supported our research:

- LLaVA: Large Language-and-Vision Assistant — [LLaVA](https://github.com/haotian-liu/LLaVA)
- LLaVA-NeXT: Next-generation multi-modal assistant — [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- lmms-eval: A standardized evaluation framework for Large Multimodal Models — [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
- Megatron-LM: Efficient, scalable training for large language models — [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- Qwen2.5-VL: Strong vision-language foundation model — [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- InternVL: Open-source large-scale vision-language foundation model — [InternVL](https://github.com/OpenGVLab/InternVL)
- Qwen3: Next-generation Qwen LLM — [Qwen](https://github.com/QwenLM/Qwen)
- MetaCLIP: Scalable contrastive pretraining — [MetaCLIP](https://github.com/facebookresearch/MetaCLIP)
- FineVision: Open Data Is All You Need — [FineVision](https://huggingface.co/spaces/HuggingFaceM4/FineVision)
