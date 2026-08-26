# 动态打包流程 (scheme-D dynamic budget)

> 面向长视频 master 帧库的离线打包:per-sample 动态分配视觉 token 预算,让每个视频尽量吃满一个 bin,并用 47m 单图精确填缝,最终产出 energon 格式的 webdataset。

- **驱动脚本**: `run_pack_master_47m.sh`
- **配置**: `dynamic_budget_default.yaml`(64k) / `dynamic_budget_128k.yaml`(长视频 128k)
- **核心脚本**: `s0_resample_from_master.py`
- **目录**: `offline_packing/` (本目录)

---

## 0. 核心思想

与"每条固定分辨率 / 固定帧数"的静态打包不同,scheme-D **按可复现随机种子为每个视频动态采一组 (分辨率, fps)**,据此决定抽多少帧、每帧多大,使单个视频的视觉 token 尽量逼近 bin 容量;同时**文本感知 (text-aware)** 地为 caption 预留 token,并用 47m 单图填满剩余空间。

关键量:
- `CAP`  : bin 序列长度(默认 `64000`;长视频用 `128000`)
- `MAXP` : bin 的 patch 数上限 = `608822 * CAP / 64000`

---

## 1. 前置:decode-once 1fps master 帧库

所有视频**只解码一次**,按 `master_fps=1` 抽全部帧,存成 content-md5 寻址的帧库
(`<root>/fps1/<md5[:2]>/<md5[2:4]>/<md5>/frame_XXXXXX.jpg + master_meta.json`)。
`master_meta.json` 含 `native_fps`(真实视频 fps)、`num_frames`(=时长秒数,因 1fps)等。
动态打包在此基础上"重采样",不重新解码视频。

---

## 2. 阶段链(7 步,顺序执行)

### s0 — 动态重采样  ★核心 (`s0_resample_from_master.py`)
对每个视频:
1. **加权随机采样** (RNG = f(seed, 样本id) → 随机但可复现):
   - 分辨率 ∈ {150k, 200k, 250k, 300k} px,权重 {.30, .30, .25, .15}
   - fps ∈ {1.0, 0.7, 0.5, 0.33},权重 {.40, .30, .20, .10}(上限 1.0 = master fps)
2. **文本预算**: `text_tok ≈ ceil(caption字符数 / chars_per_token_est=3.0)`(保守,绝不低估)
3. **视觉预算 (text-aware)**:
   ```
   visual_budget = min( budget_tokens=55000,
                        hard_total=64000 − text_tok − reserve=2000 )
   ```
   → 长 caption 自动压缩视觉预算,避免 视觉+文本 冲破 64k(否则 s2 会丢样本)
4. **帧数**: `frames = round(fps × 时长)`,被 visual_budget 封顶
   (每帧 token = 分辨率 / (patch_size² × factor²) = 分辨率 / 2304),
   夹在 `[min_frames=200, max_frames=960]`,取 `temporal_grid=2` 的偶数倍
5. 从 master 帧**等距抽 N 帧**;首个 user turn 的 `<video>` 展开为 `<image>×N`;
   写出真实 `fps`(native_fps,供训练注入时间戳)、`patch_positions` 等
- 输出: `s0_expanded.jsonl`(每视频帧数/分辨率已定死)

### s1 — 建 manifest (`s1_build_manifest.py`)
转轻量清单(路径 + 对话 + patch),并归一化对话 schema(`from/value → role/content`)。
- 输出: `manifest.jsonl`

### s2 — 算 token 长度 (`s2_compute_token_lengths.py`)
用 OV2 processor 精确计算每条 token 数(图 + 文)。
- 输出: `token_info.txt`(`name:tokens:patches`)

### s3 — 装箱 (`s3_bin_packing.py`, BFD, 1 视频/bin)
`--max-samples 1 --algorithm bfd --capacity $CAP`:
每个 bin 只放 **1 个视频**(单视频已 ~55k,接近 64k),BFD 装箱并**故意留 slack**(~9k)给下一步。
- 输出: `bins_64klen.pkl`

### s3b — 47m 填缝  ★ (`s3b_topup_with_47m.py`)
往每个 bin 剩余空间随机塞 **47m 单图**至 `CAP`:
- 池: `<47m_pool>/47m_part*/token_info.txt` (单图池, 由 --pool-tokeninfo-glob 指定)
- `--pool-shard-index/--pool-shard-count`: **disjoint 分片**,多机/多批打包各取不相交子集,防 47m 重复
- `--pool-blocklist`: 跨批去重;`--image-only`;`--max-patches $MAXP`;`--max-samples 200`
- 输出: `bins_topup.pkl`(视频 + 47m 混合)

### combine — 合并 manifest (`tools/build_combined_manifest.py`)
视频 manifest + 选中的 47m manifest → 一份合并清单,供 s4 取字节。
- 输出: `manifest_combined.jsonl`

### s4 — 生成 webdataset  ★ (容器内, `s4_bins_to_webdataset.py`)
`docker run llava_megatron:26.05` **在容器里**执行:
- 按 bin 读帧/图字节 → B 模式降采样(`--longest-edge 3072`)→ 写 tar
- **容器内 energon** 生成 `.tar.idx` + `.info.yaml`(energon 训练加载必需)
- docker 挂载必须含 `<data_root>`(含帧库与单图源盘; docker -v 按需挂载)
- 输出: `webdataset/`(`*.tar` + `*.tar.idx` + `.nv-meta/{.info.yaml, split.yaml, dataset.yaml, sample_loader.py}`)

---

## 3. 配置参考 (`dynamic_budget_default.yaml`, 64k)

```yaml
mode: dynamic
resolution:                 # min_pixels = max_pixels,按权重采
  values:  [150000, 200000, 250000, 300000]
  weights: [0.30, 0.30, 0.25, 0.15]
fps:                        # 上限 1.0(与 master 1fps 一致)
  values:  [1.0, 0.7, 0.5, 0.33]
  weights: [0.40, 0.30, 0.20, 0.10]
budget_tokens: 55000        # 单视频视觉 token 上限(留 slack 给 47m + text)
min_frames: 200             # 帧数下限
max_frames: 960             # 帧数硬上限(通常预算先封顶)
patch_size: 16
factor_multiplier: 3        # tok/帧 = 分辨率 / (patch² × factor²) = 分辨率 / 2304
min_pixels_floor: 3136      # 56×56,smart_resize 下限
temporal_grid: 2            # 帧数取偶数(temporal_grid 的倍数)
seed: 20260723              # 基种子;每样本 RNG = f(seed, 样本id) → 随机可复现
hard_total_tokens: 64000    # bin 硬上限
topup_safety_reserve: 2000  # 留给 47m topup + 安全余量
chars_per_token_est: 3.0    # 文本 token ≈ ceil(字符数/3.0),保守
```
> 128k 长视频用 `dynamic_budget_128k.yaml`:`budget_tokens=110000`, `hard_total=128000`, `max_frames=1536`。

---

## 4. 调用方式

```bash
bash run_pack_master_47m.sh <input_master_jsonl> <output_base> \
     dynamic_budget_default.yaml <shard_idx> <shard_cnt>
```
环境旋钮:
- `CAP`         : bin 长(默认 64000;长视频 128000)
- `S2_WORKERS`  / `S4_WORKERS` / `S4_CPUS` : 各阶段并行度(借用训练机时调低)
- `BLOCKLIST`   : 47m 跨批去重名单
- `<shard_idx> <shard_cnt>` : 47m 池 disjoint 分片(多机/多批打包时各机不同)

---

## 5. 与单图链 (auto_pipe.sh) 的区别

| | 动态链 (scheme-D) | auto_pipe(单图) |
|---|---|---|
| s0 重采样 | ✅ 动态帧/分辨率 | ❌ 无(单图无需) |
| 装箱 | 1 视频/bin + 47m 填缝 | 多样本/bin |
| s4 | **容器内**,出 `.tar.idx` | 裸跑,**无 idx**(需事后 `energon prepare` 补) |
| 适用 | 长视频 master | 单图 SFT / caption |

---

## 6. 一句话流程

```
decode-once 1fps master
  → s0  动态重采样 (text-aware 定帧数/分辨率,可复现随机)
  → s1  manifest
  → s2  token 长度
  → s3  BFD 每视频独占 bin(留缝)
  → s3b 47m 不相交填缝
  → combine 合并清单
  → s4  容器内编码 + energon 索引
  → 64k webdataset (tar + idx + nv-meta)
```
