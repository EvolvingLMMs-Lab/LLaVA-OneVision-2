<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/llava_onevision_2_black.svg">
    <source media="(prefers-color-scheme: light)" srcset="asset/llava_onevision_2_white.svg">
    <img alt="LLaVA-OneVision-2.0" src="asset/llava_onevision_2_white.svg" width="820" style="max-width: 100%;">
  </picture>
</p>

<p align="center">
  <strong>Fully Open Framework for Democratized Multimodal Training</strong>
</p>



<p align="center">
  🤗 <b>2.0 <a href="#models">Models</a></b> · <b><a href="#datasets">Datasets</a></b> · <b><a href="#citation">Technical Report</a></b>
</p>

<p align="center">
  🤗 <b>1.5 <a href="https://huggingface.co/collections/lmms-lab/llava-onevision-15-68d385fe73b50bd22de23713">Models</a></b> · <b><a href="https://huggingface.co/collections/lmms-lab/llava-onevision-15-68d385fe73b50bd22de23713">Datasets</a></b> · <b><a href="https://arxiv.org/abs/2509.23661">Technical Report</a></b>
</p>

---

## NEWS
- 2026-04-30: Released LLaVA-OneVision-2.0 — next-generation multimodal model, with new [LLaVA-OneVision-2.0-VideoCaption](#datasets) and [LLaVA-OneVision-2.0-Spatial](#datasets) datasets.
- 2026-02-10: Released [OneVision-Encoder](https://huggingface.co/collections/lmms-lab-encoder/onevision-encoder-6978aeb2bbe1aa13fad12d4c) — codec-aligned vision encoders, with [Technical Report](https://arxiv.org/abs/2602.08683).
- 2025-12-11: Released RL recipe for LLaVA-OneVision-1.5, with [Project](https://mvp-ai-lab.github.io/LLaVA-OneVision-1.5-RL/), [Code](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5-RL), [Data](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-RL-Data), and [Model](https://huggingface.co/mvp-lab/LLaVA-OneVision-1.5-8B-RL).
- 2025-09-30: Released the LLaVA-OneVision-1.5 [Technical Report](https://arxiv.org/abs/2509.23661).


## Contents
<!-- TOC -->
- [Introduction](#introduction)
- [Results](#evaluation-results)
- [Method](#method)
- [Models](#models)
- [Datasets](#datasets)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [LLaVA-OneVision-1.5](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-1.5/tree/1.5)


## Introduction

**LLaVA-OneVision-2.0** is the next-generation release of the LLaVA-OneVision family — a fully open 8B multimodal model that unifies image, long-form video, and spatial understanding under a single architecture, with the entire pipeline (data, encoders, training, checkpoints, logs) released end-to-end.

### 🎬 Codec-Aligned Vision Encoders

Forget uniform patchification. **OneVision-Encoder** and **OneVision-Encoder-Lang** are HEVC-style vision transformers that treat video like a codec stream — selecting only motion- and residual-rich patches and sampling dense frames sparsely instead of sparse frames densely. The result is dramatically longer temporal coverage under the same token budget, where prior ViT backbones simply run out of context.

### 🧊 One Model, Every Modality

Most open multimodal models still live in a 2D, single-image world. **LLaVA-OneVision-2.0-8B-Instruct** breaks out of it — one model, native resolution, no task-specific adapters, no hidden tricks.

- **Long video** — multi-frame reasoning with efficient codec-aligned inference
- **3D-aware spatial reasoning** — depth, layout, object relations
- **Documents, OCR, charts** — structured visual inputs at native resolution

New open-source SOTA across a broad suite of multimodal benchmarks.

### 🚀 Fully Open, Reproducible from Day One

Four datasets ship with the LLaVA-OneVision family — two new for 2.0, two carried forward from 1.5:

- **LLaVA-OneVision-2.0-VideoCaption** — extremely dense video captions
- **LLaVA-OneVision-2.0-Spatial** — 3D-aware spatial reasoning
- **LLaVA-OneVision-1.5-Mid-Training-85M** — 85M concept-balanced mid-training corpus
- **LLaVA-OneVision-1.5-Instruct** — full instruction-tuning mixture

And unlike most "open" releases, *everything* ships alongside them: encoder weights, training code, configs, and full training logs. Reproducible end to end.

## Evaluation Results

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/llava_onevision2_performance_dark_anim.svg">
    <source media="(prefers-color-scheme: light)" srcset="asset/llava_onevision2_performance_light_anim.svg">
    <img src="asset/llava_onevision2_performance_light_anim.svg" width="100%" alt="LLaVA-OneVision-2.0 Benchmark Comparison">
  </picture>
</p>

<p align="center">
  <img src="asset/method_codec_vs_frame.svg" width="100%" alt="Figure 4: codec-aligned sampling compared with uniform frame sampling across video benchmarks">
</p>


## Method

### Codec-Style Patch Selection

<p align="left">
  <img src="asset/method_codec_selection.svg" alt="Codec-Style Patch Selection: same 54-token budget, 3× more temporal range than uniform sampling" width="88%">
</p>

Standard video pipelines uniformly sample a handful of frames and process **every** patch — most of it static background. We borrow from HEVC: keep **I-frames** dense, keep only **motion- and residual-rich patches** from **P-frames**. Same 54-token budget, **18 frames** instead of 6 — 3× the temporal range, no extra LLM context, no input-type adapters.

### One Encoder, Every Modality

<p align="left">
  <img src="asset/method_unified_encoder.svg" alt="Multi-modal vision input: image, uniform frames, or codec-aligned tokens all feed the same OneVision-Encoder with shared (t, h, w) positions" width="66%">
</p>

Most multimodal stacks ship a different tokenizer per input type — one path for images, another for video, a third for multi-image. We don't. **Image, uniform frames, and codec-aligned tokens** all flow into the **same OneVision-Encoder** under a shared `(t, h, w)` position scheme. No task-specific tokenizers, no per-modality routing.

### Four-Stage Training Curriculum

We train LLaVA-OneVision-2.0 in four compact stages:

1. **Bootstrap video ability** from LLaVA-OneVision-1.5 with short 30s video captions.
2. **Instruction tune** with large multimodal instruction data and 30–180s video captions.
3. **Extend to long videos** with 10–15 min captions and public video instruction data.
4. **Refine codec, spatial, and tracking skills** with denser long-video sampling, point tracking, and 4M spatial samples.

The curriculum mixes LLaVA-OneVision-1.5 data, FineVision, and new in-house video caption/spatial datasets; we do not synthesize any video instruction data.


## Models

| Model                           | HF Link                                                                                               | Training Log                                                                                 |
| ------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| LLaVA-OneVision-2.0-8B-Instruct | —                                                                                                     | —                                                                                            |
| LLaVA-OneVision-2.0-4B-Instruct | —                                                                                                     | —                                                                                            |
| LLaVA-OneVision-1.5-4B-Instruct | [🤗 HF / 4B-Instruct](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct)                 | [📈 TensorBoard](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-4B-Instruct/tensorboard) |
| LLaVA-OneVision-1.5-8B-Instruct | [🤗 HF / 8B-Instruct](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct)                 | [📈 TensorBoard](https://huggingface.co/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct/tensorboard) |
| OneVision-Encoder               | [🤗 HF / OneVision-Encoder](https://huggingface.co/lmms-lab-encoder/onevision-encoder-large)           | —                                                                                            |
| OneVision-Encoder-Lang          | [🤗 HF / OneVision-Encoder-Lang](https://huggingface.co/lmms-lab-encoder/onevision-encoder-large-lang) | —                                                                                            |

## Datasets

| Description                          | Link                                                                                                   | Status    |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------ | --------- |
| LLaVA-OneVision-2.0-VideoCaption     | [🤗HF / VideoCaption](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-2-Data/tree/main/mid_training_video) | Available |
| LLaVA-OneVision-2.0-Spatial          | [🤗HF / Spatial](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-2-Data/tree/main/spatial)     | Available |
| LLaVA-OneVision-1.5-Mid-Training-85M | [🤗HF / Mid-Training 85M](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M) | Available |
| LLaVA-OneVision-1.5-Instruct         | [🤗HF / Instruct-Data](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)       | Available |

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="asset/llava_onevision2_data_distribution_dark_anim.svg">
    <source media="(prefers-color-scheme: light)" srcset="asset/llava_onevision2_data_distribution_light_anim.svg">
    <img src="asset/llava_onevision2_data_distribution_light_anim.svg" width="100%" alt="LLaVA-OneVision Data Distribution">
  </picture>
</p>


## Contributors
Thanks so much to all of our amazing contributors!

<!-- readme: collaborators,contributors,jiankangdeng/- -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/anxiangsir">
                    <img src="https://avatars.githubusercontent.com/u/31175974?s=80&v=4" width="80" height="80" alt="anxiangsir"/>
                    <br />
                    <sub><b>anxiangsir</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/yiyexy">
                    <img src="https://avatars.githubusercontent.com/u/35927125?s=80&v=4" width="80" height="80" alt="yiyexy"/>
                    <br />
                    <sub><b>yiyexy</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/fdcp">
                    <img src="https://avatars.githubusercontent.com/u/15667917?s=80&v=4" width="80" height="80" alt="fdcp"/>
                    <br />
                    <sub><b>fdcp</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/wideyard">
                    <img src="https://avatars.githubusercontent.com/u/101321826?s=80&v=4" width="80" height="80" alt="wideyard"/>
                    <br />
                    <sub><b>wideyard</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Lornatang">
                    <img src="https://avatars.githubusercontent.com/u/31124350?s=80&v=4" width="80" height="80" alt="Lornatang"/>
                    <br />
                    <sub><b>Lornatang</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/chengzheng345">
                    <img src="https://avatars.githubusercontent.com/u/209475443?s=80&v=4" width="80" height="80" alt="chengzheng345"/>
                    <br />
                    <sub><b>chengzheng345</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Luodian">
                    <img src="https://avatars.githubusercontent.com/u/15847405?s=80&v=4" width="80" height="80" alt="Luodian"/>
                    <br />
                    <sub><b>Luodian</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/kcz358">
                    <img src="https://avatars.githubusercontent.com/u/92624596?s=80&v=4" width="80" height="80" alt="kcz358"/>
                    <br />
                    <sub><b>kcz358</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/killTheHostage">
                    <img src="https://avatars.githubusercontent.com/u/16442720?s=80&v=4" width="80" height="80" alt="killTheHostage"/>
                    <br />
                    <sub><b>killTheHostage</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/mathCrazyy">
                    <img src="https://avatars.githubusercontent.com/u/20607153?s=80&v=4" width="80" height="80" alt="mathCrazyy"/>
                    <br />
                    <sub><b>mathCrazyy</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/wkzhang636">
                    <img src="https://avatars.githubusercontent.com/u/194186498?s=80&v=4" width="80" height="80" alt="wkzhang636"/>
                    <br />
                    <sub><b>wkzhang636</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/yunglechao">
                    <img src="https://avatars.githubusercontent.com/u/7631185?s=80&v=4" width="80" height="80" alt="yunglechao"/>
                    <br />
                    <sub><b>yunglechao</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/RobitYadda">
                    <img src="https://avatars.githubusercontent.com/u/6811311?s=80&v=4" width="80" height="80" alt="RobitYadda"/>
                    <br />
                    <sub><b>RobitYadda</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/fengshikun">
                    <img src="https://avatars.githubusercontent.com/u/2499990?s=80&v=4" width="80" height="80" alt="fengshikun"/>
                    <br />
                    <sub><b>fengshikun</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/GeoffreyChen777">
                    <img src="https://avatars.githubusercontent.com/u/14183213?s=80&v=4" width="80" height="80" alt="GeoffreyChen777"/>
                    <br />
                    <sub><b>GeoffreyChen777</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/didizhu-judy">
                    <img src="https://avatars.githubusercontent.com/u/34787894?s=80&v=4" width="80" height="80" alt="didizhu-judy"/>
                    <br />
                    <sub><b>didizhu-judy</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/yshenaw">
                    <img src="https://avatars.githubusercontent.com/u/45809710?s=80&v=4" width="80" height="80" alt="yshenaw"/>
                    <br />
                    <sub><b>yshenaw</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Yangsenqiao">
                    <img src="https://avatars.githubusercontent.com/u/73487993?s=80&v=4" width="80" height="80" alt="Yangsenqiao"/>
                    <br />
                    <sub><b>Yangsenqiao</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/YunyaoYan">
                    <img src="https://avatars.githubusercontent.com/u/109638667?s=80&v=4" width="80" height="80" alt="YunyaoYan"/>
                    <br />
                    <sub><b>YunyaoYan</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/FeilongTangmonash">
                    <img src="https://avatars.githubusercontent.com/u/152372878?s=80&v=4" width="80" height="80" alt="FeilongTangmonash"/>
                    <br />
                    <sub><b>FeilongTangmonash</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Jinghao-Guo">
                    <img src="https://avatars.githubusercontent.com/u/212396229?s=80&v=4" width="80" height="80" alt="Jinghao-Guo"/>
                    <br />
                    <sub><b>Jinghao-Guo</b></sub>
                </a>
            </td>
		</tr>
	</tbody>
</table>
<!-- readme: collaborators,contributors,jiankangdeng/- -end -->

## Citation

If you find *LLaVA-OneVision-1.5* useful in your research, please consider to cite the following related papers:

```
@inproceedings{LLaVA-OneVision-2.0,
  title={LLaVA-OneVision-2.0},
  author={llava-onevision contributors},
  booktitle={arXiv},
  year={2026}
}

@inproceedings{LLaVA-OneVision-1.5,
  title={LLaVA-OneVision-1.5: Fully Open Framework for Democratized Multimodal Training},
  author={An, Xiang and Xie, Yin and Yang, Kaicheng and Zhang, Wenkang and Zhao, Xiuwei and Cheng, Zheng and Wang, Yirui and Xu, Songcen and Chen, Changrui and Wu, Chunsheng and Tan, Huajie and Li, Chunyuan and Yang, Jing and Yu, Jie and Wang, Xiyao and Qin, Bin and Wang, Yumeng and Yan, Zizhen and Feng, Ziyong and Liu, Ziwei and Li, Bo and Deng, Jiankang},
  booktitle={arXiv},
  year={2025}
 }

@article{tang2026onevisionencoder,
  title={OneVision-Encoder: Codec-Aligned Sparsity as a Foundational Principle for Multimodal Intelligence},
  author={Tang, Feilong and An, Xiang and Yan, Yunyao and Xie, Yin and Qin, Bin and Yang, Kaicheng and Shen, Yifei and Zhang, Yuanhan and Li, Chunyuan and Feng, Shikun and Chen, Changrui and Tan, Huajie and Hu, Ming and Zhang, Manyuan and Li, Bo and Feng, Ziyong and Liu, Ziwei and Ge, Zongyuan and Deng, Jiankang},
  journal={arXiv preprint arXiv:2602.08683},
  year={2026}
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
