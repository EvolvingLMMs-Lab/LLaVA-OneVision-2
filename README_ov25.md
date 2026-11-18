# LLaVA-OneVision-2.5 × Qwen3.5-4B 混合注意力基座 · 使用说明

> 本文件讲清楚 **HF 权重 ↔ Megatron-Core(mcore)权重的双向转换**,以及权重构建、训练、和几处必须知道的坑。所有命令均取自本仓库实际脚本,路径可直接替换使用。

---

## 0. 这是什么

把一个 **raw ViT**(p16 / spatial_merge=3)和从 **Qwen3.5-4B VLM** 抽出的 **hybrid-attention LLM**(线性 GatedDeltaNet + 每 4 层一次全注意力,3D M-RoPE,partial rotary 0.25)组合成 OV2.5 模型,并适配 AIAK-Megatron 训练。

与原 Qwen3 基座的差异,核心是:**Qwen3.5 是混合注意力 + M-RoPE + 词表 248320,vision token id 也全变了**,所以转换/训练都要走专用代码路径。

---

## 1. 环境 / 镜像

- 训练镜像:`llava_megatron:26.07`（已把 **FLA (flash-linear-attention 0.5.1)** 烤进去；GatedDeltaNet 的 triton kernel 依赖它）。
- 转换/训练都在容器内跑：容器 `llava_megatron_container`，把放权重和数据的盘挂进去（本文档统一写成 `/models` 和 `/data`），`--network host --ipc host --privileged --gpus all`。
- 首次运行会**冷编译 fla triton kernel（约 5–15 分钟）**，首个 iteration 特别慢属正常，不是卡死。
- triton 3.2 低于推荐 3.3，只有 warning，可用。

---

## 2. 权重构建总览

```
raw ViT  ┐
         ├─(A) merge_ov2 ─→  合并 HF 模型  ─(B) hf→mcore ─→  mcore 权重(训练用)
Qwen3.5 LLM ┘                     ▲                                  │
                                  └────────(C) mcore→hf ◀────────────┘ (训练产物回 HF)
```

- 合并 HF 模型：`/models/llava_onevision2_4b_ov25_p16m3`
- mcore 权重（TP 1/2/4）：`.../llava_onevision2_4b_ov25_p16m3_mcore_tp{1,2,4}pp1`

---

## 3. (B) HuggingFace → Megatron-Core   ★主流程

脚本：`examples/llava_onevision2/convert/convert_4b_ov25_hf_to_mcore.sh`

```bash
# 在容器内、仓库根目录执行
bash examples/llava_onevision2/convert/convert_4b_ov25_hf_to_mcore.sh \
     <LOAD_HF>  <SAVE_MCORE>  [TP]  [PP]

# 例（TP=1）
bash examples/llava_onevision2/convert/convert_4b_ov25_hf_to_mcore.sh \
  /models/llava_onevision2_4b_ov25_p16m3 \
  /models/llava_onevision2_4b_ov25_p16m3_mcore_tp1pp1  1  1
```

参数：`LOAD_HF`=合并好的 HF 模型；`SAVE_MCORE`=输出 mcore 目录；`TP∈{1,2,4}`，`PP=1`。

### 内部做了什么（四路各转再合并）

| 组件 | 转换器 | 说明 |
|---|---|---|
| **language(Qwen3.5)** | `custom/llava_onevision2/ov25_hf_to_mcore.py` | **专用**转换器（不走通用 model.py+qwen3.json）。传 `--num_layers=32 --full_attention_interval=4 --tensor_model_parallel_size=$TP` |
| ViT | 通用 `tools/convert_checkpoint/model.py` | 复用 Qwen3-4B 同款(p16/merge3 ViT 一致) |
| adapter | `custom/llava_onevision2/adapter.py` | 复用 |
| vision patch | `custom/llava_onevision2/vision_patch.py` | 复用 |
| **合并** | `custom/llava_onevision2/merge_megatron.py` | 四路 `release/` 合并成一个 mcore ckpt，写 `latest_checkpointed_iteration.txt=release` |

### 为什么 language 要专用转换器
Qwen3.5 的线性注意力层 `in_proj_qkvz` / `in_proj_ba` 是**多 head 的 q/k/v/z 交错打包**布局，需按 head 组重排；SwiGLU 的 `fc1` 要 transpose 后按 TP 切；线性注意力的 key/value 头在 TP≥2 时也要沿 TP 切。这些逻辑在经过验证的 `aiak_training_llm/models/ov25/hf_convert.py` 里，专用转换器基于它。

---

## 4. (C) Megatron-Core → HuggingFace   ★主流程

脚本：`examples/llava_onevision2/convert/convert_4b_ov25_mcore_to_hf.sh`

```bash
bash examples/llava_onevision2/convert/convert_4b_ov25_mcore_to_hf.sh \
     <LOAD_MCORE>  <SAVE_HF>  [TP]  [PP]  [REFERENCE]

# 例（从训练产物 TP1 转回 HF）
bash examples/llava_onevision2/convert/convert_4b_ov25_mcore_to_hf.sh \
  ./checkpoints/ov25_stage_vit_adapter_packed_muon/iter_0002000 \
  /models/ov25_stage_vit_adapter_from_mcore  1  1
```

参数：`LOAD_MCORE`=mcore ckpt（含 `release/` 或直接含 `mp_rank_*` 的目录；也可指向训练的 `iter_XXXXXXX/`）；`SAVE_HF`=输出 HF 目录；`REFERENCE`=拷 config/modeling/processor/tokenizer 的参考 HF 模型（默认 `.../llava_onevision2_4b_ov25_p16m3`，**只重生成权重，其余文件从参考拷**）。

### 内部（B 的逆过程）
- language 用**专用逆转换器** `ov25_mcore_to_hf.py`；ViT / adapter / vision_patch 复用通用逆转换器；`merge_huggingface.py` 合并成一个 `LlavaOnevision2ForConditionalGeneration`。
- **args 注入小机关**:ViT 的逆转换(model.py)需要 ckpt 里带 megatron `args`。真实训练 ckpt 都有；但**前向 language 转换器写出的 ckpt 没有 `args`**。脚本会自动探测,缺失时用参考模型跑一次「throwaway 前向 vision 转换」合成 args,再用 `ov25_inject_args.py` 注入到一份工作副本（**绝不改原 ckpt**）。所以从「hf→mcore 产物」直接转回时也能工作。

### 往返一致性
`HF → mcore → HF` 已验证**逐比特一致**；整模型前向 cos 相似度 0.9999997。

---

## 5. (A) 合并 HF 模型（merge_ov2，参考）

如需从零重建合并 HF 模型：
```bash
python -m transformers_impl.merge_ov2 merge \
  --variant dense \
  --vit   <raw_vit_path> \
  --llm   <qwen3.5_llm_path> \
  --processor <ov25_processor> \
  --spatial-merge-size 3 --target-dtype bf16 \
  --out   /models/llava_onevision2_4b_ov25_p16m3
```
注意 **`--spatial-merge-size 3`**(p16m3)。合并后 48 个 SSM 参数(`A_log`、线性注意力 `norm`)会保持 **fp32**(其余 bf16),不可全降 bf16。

---

## 6. 训练脚本

| 脚本 | 用途 |
|---|---|
| `examples/.../quick_start_4b/ov25_stage_1_alignment_adapter_only.sh` | stage-1 对齐：非 packed、仅 adapter、adam、TP1、seq32k、GBS256、lr2e-5 |
| `examples/.../quick_start_4b/ov25_stage_vit_adapter_packed_muon.sh` | 续训练：**packed + muon**，`--trainable-modules adapter vision_model`，`--finetune` 从 step 0 载 stage-1 权重 |
| `examples/.../quick_start_4b/ov25_packed64k_tp4_sp.sh` | TP4+SP+packed 64k 模板 |

启动（双机示例，容器内）：
```bash
# 每台节点各执行（脚本按本机 IP 自动定 node_rank）
docker exec -w /workspace/LLaVA-OneVision-2.5 llava_megatron_container \
  bash examples/llava_onevision2/quick_start_4b/ov25_stage_vit_adapter_packed_muon.sh
```
默认节点 `LIST_IP="192.0.2.11 192.0.2.12"`，可用环境变量覆盖：`LIST_IP="ip1 ip2 ..."`。

关键 CLI（Qwen3.5 专有,脚本已内置）：
```
--model-name llava-onevision2-4b-ov25
--linear-key-head-dim 128 --linear-value-head-dim 128
--linear-num-key-heads 16 --linear-num-value-heads 32
--linear-conv-kernel-dim 4 --full-attention-interval 4
--mrope-section 11 11 10 --mrope-interleaved --partial-rotary-factor 0.25
--vision-frame-windows-size 1
--cross-entropy-loss-fusion          # 词表 248320 大，TP1 下省显存
--hf-tokenizer-path /models/ov25_processor
```

---

## 7. 必须知道的坑

1. **vision token id**：Qwen3.5 是 image `248056` / video `248057` / start `248053` / end `248054`（Qwen3 是 151655/…）。任何硬编码 151655 的地方会导致「图像没被识别 → adapter 无梯度、grad norm=0」。已在 provider 与 `get_batch` 修正（按 `language_model_type` 选 id）。
2. **packed M-RoPE(用真实 patch_positions,对齐 Qwen3.5)**：Qwen3.5 的 temporal = **帧序号** arange×1(非秒、非源帧号),真实时序由 `<X.X seconds>` 时间戳文本承载(数据已插)。`get_rope_index_packed` 用 `patch_position.npy`(已铺到 forward、编码器已 block-layout)降到 per-merged-token `(t, h//merge, w//merge)`：temporal=每视觉体内真实 t 的 dense-rank(0/30/60→0/1/2),h/w=merged 网格,每视觉体共享 base,文本顺序位置。断言 `总patch/9==image_pad数`。patch_positions 缺失时回退到 grid 展开路径。
3. **续训 step 归零**：用 `--finetune`（只载权重、iteration=0、fresh 优化器/LR），而非仅 `--no-load-optim --no-load-rng`（后者会继承 stage-1 的 iteration 计数）。
4. **TP 与 ckpt 匹配**：训练的 `--load` 必须用同 TP 的 mcore 权重。TP>1 时脚本会自动加 `--sequence-parallel`。
5. **SSM 参数 fp32**：合并/转换全程保持 48 个 SSM 参数 fp32。
6. **多节点 NCCL**：**所有节点用同一 26.07 镜像(auto-detect NCCL,不强制 IB)**,脚本不设 NCCL env。
7. **packed 数据免重打**：offline_packing 存的是原始 text+image,加载时**实时 tokenize**,所以换 Qwen3.5 tokenizer 无需重新打包。

---

## 8. 文件地图

```
aiak_training_llm/models/ov25/            # 新增 mcore 基座模块
  gated_deltanet.py    线性注意力层(FLA delta 递推)
  gated_attention.py   全注意力层(带输出门)
  rmsnorm.py           Qwen3.5 RMSNorm
  ov25_rope.py       M-RoPE(3D/interleaved/partial)
  ov25_layer_spec.py 按 full_attention_interval 组装混合层栈
  hf_convert.py        HF↔mcore 权重映射数学(转换器基于它)
aiak_training_llm/models/llava_onevision2/
  llava_onevision2_config.py    arch: llava_onevision2_4b_ov25()
  llava_onevision2_model.py     get_rope_index / get_rope_index_packed
  llava_onevision2_provider.py  build 模型 + vision token id
tools/convert_checkpoint/custom/llava_onevision2/
  ov25_hf_to_mcore.py / ov25_mcore_to_hf.py / ov25_inject_args.py
examples/llava_onevision2/convert/          # 转换 wrapper(见 §3/§4)
examples/llava_onevision2/quick_start_4b/   # 训练脚本(见 §6)
```

---
*内容基于本仓库实际代码与脚本。*
