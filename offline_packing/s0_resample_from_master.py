#!/usr/bin/env python3
"""s0_resample_from_master.py -- PACK-TIME resample from DECODE-ONCE master frames.

Two modes:
  * DYNAMIC (--config <yaml>): per-sample sample (resolution, fps) from weighted
    distributions (reproducible per-sample RNG; NOT a rigid duration->budget map),
    derive frames = round(fps * duration) capped by a token budget. Gives frame/res
    diversity. 47m topup later fills the leftover slack.
  * FIXED (--max-frames N --max-pixels M): every master sample uses the same N/M.

For each MASTER sample: read master_meta -> pick (res, frames) -> temporal subset ->
smart_resize to res (48-aligned, uniform per video) -> patch_positions (VERBATIM algo
from run_cut_frames, cached at <master_dir>/derived/mp<MP>_mf<MF>/) -> expand <video>
into N <image> -> emit expanded line (images/patch_positions/resize_hw/is_master + dyn provenance).
LEGACY samples (no master_dir) pass through unchanged.
"""
from __future__ import annotations
import argparse, hashlib, json, logging, math, os
import numpy as np


# ---- copied VERBATIM from run_cut_frames.py (byte-identical patch_positions) ----
def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if max_pixels and h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif min_pixels and h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def build_patch_positions_for_frame(frame_index, height, width, patch_size, min_pixels, max_pixels, factor_multiplier=2):
    resized_h, resized_w = smart_resize(height, width, factor=patch_size * factor_multiplier,
                                        min_pixels=min_pixels, max_pixels=max_pixels)
    tokens_h = resized_h // patch_size
    tokens_w = resized_w // patch_size
    frame_tokens = tokens_h * tokens_w
    per = np.arange(frame_tokens, dtype=np.int64)
    return np.stack([np.full((frame_tokens,), frame_index, dtype=np.int64), per // tokens_w, per % tokens_w], axis=1)
# --------------------------------------------------------------------------------


def select_positions(n, target):
    if target >= n:
        return list(range(n))
    if target <= 0:
        return []
    return [round(i * (n - 1) / (target - 1)) if target > 1 else 0 for i in range(target)]


def expand_video_to_images(messages, num_images):
    for m in messages:
        c = m.get("content")
        if isinstance(c, str) and ("<image>" in c or "<video>" in c):
            m["content"] = (c.replace("<image>\n", "").replace("<image>", "")
                             .replace("<video>\n", "").replace("<video>", "")).lstrip("\n")
    block = "".join(["<image>\n"] * num_images)
    for m in messages:
        if m.get("role") == "user":
            m["content"] = block + m.get("content", "")
            break


def load_meta(master_dir):
    with open(os.path.join(master_dir, "master_meta.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def duration_sec(meta):
    mfps = meta.get("master_fps", 1.0) or 1.0
    return meta.get("num_frames", 0) / mfps


# ---- dynamic sampling ----
def _seeded_rng(base_seed, key):
    import random
    h = hashlib.md5(f"{base_seed}:{key}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _wchoice(rng, values, weights):
    r = rng.random() * sum(weights); c = 0.0
    for v, w in zip(values, weights):
        c += w
        if r <= c:
            return v
    return values[-1]


def caption_char_len(messages):
    """Total caption chars (placeholders stripped) -- proxy for text tokens."""
    total = 0
    for m in messages or []:
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        if isinstance(c, str):
            total += len(c.replace("<image>", "").replace("<video>", ""))
    return total


def sample_res_frames(cfg, dur_s, rng, text_tok_est=0, n_videos=1):
    res = _wchoice(rng, cfg["resolution"]["values"], cfg["resolution"]["weights"])
    fps = _wchoice(rng, cfg["fps"]["values"], cfg["fps"]["weights"])
    tpf = res / (cfg["patch_size"] ** 2 * cfg["factor_multiplier"] ** 2)   # MP/2304 (merged tok/frame)
    # text-aware budget: hard 64k cap - caption text - topup/safety reserve, split across videos.
    hard_total = int(cfg.get("hard_total_tokens", 64000))
    reserve = int(cfg.get("topup_safety_reserve", 2000))
    vis_total = max(0, hard_total - int(text_tok_est) - reserve)
    vis_budget = min(int(cfg["budget_tokens"]), vis_total // max(1, int(n_videos)))
    frames = round(fps * dur_s)
    frames = min(frames, int(vis_budget // tpf), int(cfg.get("max_frames", 960)))
    frames = max(int(cfg["min_frames"]), frames)
    g = int(cfg.get("temporal_grid", 1))
    if g > 1 and frames % g:
        frames -= frames % g
    return res, frames, fps


def resample_one_video(master_dir, meta, max_pixels, max_frames, patch_size, factor_multiplier, min_pixels):
    """Return (selected_frame_paths, (th,tw), derived_npy_path) using a cached derived/ dir."""
    frames = meta.get("images") or []
    src_idx = meta.get("frame_indices") or list(range(len(frames)))
    if not frames:
        raise ValueError(f"master has no frames: {master_dir}")
    from PIL import Image
    with Image.open(frames[0]) as im:
        mw, mh = im.size
    th, tw = smart_resize(mh, mw, factor=patch_size * factor_multiplier, min_pixels=min_pixels, max_pixels=max_pixels)
    pos = select_positions(len(frames), min(len(frames), max_frames))
    sel_frames = [frames[p] for p in pos]
    sel_t = [int(src_idx[p]) for p in pos]

    derived = os.path.join(master_dir, "derived", f"mp{max_pixels}_mf{max_frames}")
    npy_path = os.path.join(derived, "patch_positions.npy")
    sel_json = os.path.join(derived, "sel.json")
    if os.path.exists(npy_path) and os.path.exists(sel_json):
        return sel_frames, (th, tw), npy_path
    os.makedirs(derived, exist_ok=True)
    pp = [build_patch_positions_for_frame(t, mh, mw, patch_size, min_pixels, max_pixels, factor_multiplier) for t in sel_t]
    arr = np.concatenate(pp, axis=0) if pp else np.zeros((0, 3), dtype=np.int64)
    tmp = npy_path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, npy_path)
    with open(sel_json, "w", encoding="utf-8") as f:
        json.dump({"frames": [os.path.basename(x) for x in sel_frames], "target_hw": [th, tw],
                   "num_frames": len(sel_frames), "max_pixels": max_pixels, "max_frames": max_frames}, f)
    return sel_frames, (th, tw), npy_path


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--config", default="", help="dynamic-budget yaml; if given, dynamic mode (ignores --max-frames/--max-pixels)")
    ap.add_argument("--max-frames", type=int, default=-1, help="required in fixed mode")
    ap.add_argument("--max-pixels", type=int, default=-1, help="required in fixed mode")
    ap.add_argument("--patch-size", type=int, default=16)
    ap.add_argument("--factor-multiplier", type=int, default=3)
    ap.add_argument("--min-pixels", type=int, default=56 * 56)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logging.info("dynamic mode: config=%s  res=%s fps=%s budget=%d",
                     args.config, cfg["resolution"]["values"], cfg["fps"]["values"], cfg["budget_tokens"])
        patch = int(cfg.get("patch_size", 16)); fmult = int(cfg.get("factor_multiplier", 3))
        minpix = int(cfg.get("min_pixels_floor", 56 * 56)); base_seed = cfg.get("seed", 0)
    else:
        patch = args.patch_size; fmult = args.factor_multiplier; minpix = args.min_pixels

    n_master = n_legacy = n_drop = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            mdirs = s.get("master_dir")
            if not mdirs:
                n_legacy += 1
                fout.write(json.dumps(s, ensure_ascii=False) + "\n")
                continue
            if cfg is None and (args.max_frames <= 0 or args.max_pixels <= 0):
                raise SystemExit("fixed mode needs --max-frames and --max-pixels (or use --config for dynamic mode)")
            if isinstance(mdirs, str):
                mdirs = [mdirs]
            sid = str(s.get("id", lineno))
            try:
                all_frames, resize_hw, pp_paths, dyn, fps_vals = [], [], [], [], []
                cpt = float(cfg.get("chars_per_token_est", 3.0)) if cfg is not None else 3.0
                text_tok_est = int(math.ceil(caption_char_len(s.get("messages", [])) / cpt)) if cfg is not None else 0
                for vi, md in enumerate(mdirs):
                    meta = load_meta(md)
                    fps_vals.append(meta.get("native_fps"))
                    if cfg is not None:
                        rng = _seeded_rng(base_seed, f"{sid}:{vi}")
                        res, nf, fps = sample_res_frames(cfg, duration_sec(meta), rng, text_tok_est, len(mdirs))
                    else:
                        res, nf, fps = args.max_pixels, args.max_frames, None
                    sf, (th, tw), npy = resample_one_video(md, meta, res, nf, patch, fmult, minpix)
                    all_frames.extend(sf)
                    resize_hw.extend([[th, tw]] * len(sf))
                    pp_paths.append(npy)
                    dyn.append({"res": res, "frames": len(sf), "fps": round(fps, 4) if fps else None,
                                "hw": [th, tw], "dur_s": round(duration_sec(meta), 1),
                                "text_tok_est": text_tok_est})
                if not all_frames:
                    n_drop += 1; continue
                expand_video_to_images(s.get("messages", []), len(all_frames))
                fout.write(json.dumps({
                    "id": s.get("id"), "messages": s.get("messages", []),
                    "images": all_frames, "patch_positions": pp_paths, "resize_hw": resize_hw,
                    "is_master": True, "video_source": s.get("video_source"), "master_dir": mdirs,
                    "dyn": dyn,
                    "fps": (fps_vals[0] if len(fps_vals) == 1 else fps_vals),   # REAL source fps (cv2 CAP_PROP_FPS) for timestamps
                }, ensure_ascii=False) + "\n")
                n_master += 1
            except Exception as exc:
                logging.warning("s0 resample failed (line %d id=%s): %s", lineno, sid, exc)
                n_drop += 1
            if (n_master + n_legacy) % 500 == 0:
                logging.info("s0 progress: master=%d legacy=%d drop=%d", n_master, n_legacy, n_drop)
    logging.info("s0 DONE: master=%d legacy=%d drop=%d -> %s", n_master, n_legacy, n_drop, args.output_jsonl)


if __name__ == "__main__":
    main()
