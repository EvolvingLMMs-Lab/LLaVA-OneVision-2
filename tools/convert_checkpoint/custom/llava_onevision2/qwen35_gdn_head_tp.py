#!/usr/bin/env python3
"""Convert Qwen3.5 mcore checkpoints between replicated and head-parallel GDN layouts.

Replicated (default): every TP rank holds the full gated delta-net (GDN) params of each linear
layer. Head-parallel (--gdn-head-parallel): TP rank r holds only the params of key heads
[r*nk/tp, (r+1)*nk/tp) and their value heads:

  in_proj_qkvz.weight  rows, contiguous chunk r   (rows are key-head-major: [q k v z] per head)
  in_proj_ba.weight    rows, contiguous chunk r   (key-head-major: [b a] per head)
  A_log, dt_bias       contiguous chunk r         (one entry per value head)
  conv1d.weight/bias   chunk r of each of the q | k | v channel blocks
  out_proj.weight      columns, contiguous chunk r (input is head-major)
  out_norm.weight      replicated (shared by all heads)

Usage
-----
  detect  <ckpt_dir>             print "replicated" or "head_parallel"
  shard   <src_dir> <dst_dir>    replicated -> head-parallel (weights only, for --finetune /
                                 --no-load-optim; optimizer state is not carried over)
  unshard <src_dir> <dst_dir>    head-parallel -> replicated (weights only; used before HF export)

<ckpt_dir> directly contains mp_rank_TT or mp_rank_TT_PPP dirs (an iter_XXXXXXX dir or release/).
All other tensors are copied unchanged. The source is never modified.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import torch

RANK_DIR = re.compile(r"^mp_rank_(\d{2})(?:_(\d{3}))?$")
GDN = re.compile(r"^(?P<pre>.*language_model\.decoder\.layers\.\d+\.self_attention\.)"
                 r"(?P<name>in_proj_qkvz\.weight|in_proj_ba\.weight|A_log|dt_bias|conv1d\.weight|conv1d\.bias"
                 r"|out_proj\.weight|out_norm\.weight)$")
DROP_KEYS = ("optimizer", "opt_param_scheduler", "rng_state")


def _stages(ckpt_dir):
    stages = defaultdict(dict)
    for d in os.listdir(ckpt_dir):
        m = RANK_DIR.match(d)
        if m and os.path.isfile(os.path.join(ckpt_dir, d, "model_optim_rng.pt")):
            stages[m.group(2) or "000"][int(m.group(1))] = d
    if not stages:
        sys.exit(f"[gdn_head_tp] no mp_rank_*/model_optim_rng.pt under {ckpt_dir}")
    return {pp: [tps[t] for t in sorted(tps)] for pp, tps in sorted(stages.items())}


def _load(ckpt_dir, d):
    return torch.load(os.path.join(ckpt_dir, d, "model_optim_rng.pt"), map_location="cpu", weights_only=False)


def _dims(ck):
    a = ck.get("args")
    get = (lambda k, dflt: getattr(a, k, dflt)) if a is not None else (lambda k, dflt: dflt)
    return dict(nk=get("linear_num_key_heads", 16), nv=get("linear_num_value_heads", 32),
                kd=get("linear_key_head_dim", 128), vd=get("linear_value_head_dim", 128))


def _layout(ck, tp):
    """'head_parallel' if the GDN tensors are 1/tp of the full size, else 'replicated'."""
    d = _dims(ck)
    for k, v in ck["model"].items():
        m = GDN.match(k)
        if m and m.group("name") == "in_proj_ba.weight":
            return "head_parallel" if tp > 1 and v.shape[0] == 2 * d["nv"] // tp else "replicated"
    return "replicated"


def _conv_split(d, local):
    """Channel sizes of the q | k | v blocks of conv1d (full or per-rank)."""
    div = local
    return [d["nk"] * d["kd"] // div, d["nk"] * d["kd"] // div, d["nv"] * d["vd"] // div]


def _shard(name, full, r, tp, d):
    if name in ("in_proj_qkvz.weight", "in_proj_ba.weight", "A_log", "dt_bias"):
        return full.chunk(tp, 0)[r].clone()
    if name == "out_proj.weight":
        return full.chunk(tp, 1)[r].clone()
    if name.startswith("conv1d."):
        return torch.cat([b.chunk(tp, 0)[r] for b in full.split(_conv_split(d, 1), 0)], 0).clone()
    return full.clone()  # out_norm.weight


def _unshard(name, parts, tp, d):
    if name in ("in_proj_qkvz.weight", "in_proj_ba.weight", "A_log", "dt_bias"):
        return torch.cat(parts, 0)
    if name == "out_proj.weight":
        return torch.cat(parts, 1)
    if name.startswith("conv1d."):
        blocks = [p.split(_conv_split(d, tp), 0) for p in parts]  # per rank: (q, k, v)
        return torch.cat([torch.cat([b[i] for b in blocks], 0) for i in range(3)], 0)
    for p in parts[1:]:  # out_norm.weight: replicated, TP-reduced -> must match
        if not torch.equal(p, parts[0]):
            sys.exit(f"[gdn_head_tp] out_norm differs across TP ranks for {name}; refusing to unshard")
    return parts[0]


def _save(dst, d, ck):
    for k in DROP_KEYS:
        ck.pop(k, None)
    os.makedirs(os.path.join(dst, d), exist_ok=True)
    torch.save(ck, os.path.join(dst, d, "model_optim_rng.pt"))


def convert(src, dst, to):
    if os.path.abspath(src) == os.path.abspath(dst):
        sys.exit("[gdn_head_tp] dst must differ from src (source is never modified)")
    for pp, dirs in _stages(src).items():
        tp = len(dirs)
        cks = [_load(src, d) for d in dirs]
        have = _layout(cks[0], tp)
        want = "head_parallel" if to == "shard" else "replicated"
        if tp == 1:
            sys.exit("[gdn_head_tp] TP=1 checkpoint: nothing to shard")
        if have == want:
            sys.exit(f"[gdn_head_tp] stage {pp} is already {have}")
        d = _dims(cks[0])
        if d["nk"] % tp or d["nv"] % tp:
            sys.exit(f"[gdn_head_tp] heads ({d['nk']} key / {d['nv']} value) not divisible by TP={tp}")
        n = 0
        for key in list(cks[0]["model"].keys()):
            m = GDN.match(key)
            if not m:
                continue
            name = m.group("name")
            if to == "shard":
                full = cks[0]["model"][key]
                for c in cks[1:]:
                    if not torch.equal(c["model"][key], full):
                        sys.exit(f"[gdn_head_tp] replicated copies differ at {key}: run "
                                 "qwen35_tp_consistency.py resync first")
                for r, c in enumerate(cks):
                    c["model"][key] = _shard(name, full, r, tp, d)
            else:
                full = _unshard(name, [c["model"][key] for c in cks], tp, d)
                for c in cks:
                    c["model"][key] = full.clone()
            n += 1
        for c in cks:
            if c.get("args") is not None:
                c["args"].gdn_head_parallel = to == "shard"
        for dname, c in zip(dirs, cks):
            _save(dst, dname, c)
        print(f"[gdn_head_tp] stage {pp}: {have} -> {want}, {n} GDN tensors x {tp} TP ranks")
    print(f"[gdn_head_tp] wrote {dst} (weights only)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("detect").add_argument("ckpt_dir")
    for c in ("shard", "unshard"):
        p = sub.add_parser(c)
        p.add_argument("src")
        p.add_argument("dst")
    a = ap.parse_args()
    if a.cmd == "detect":
        dirs = next(iter(_stages(a.ckpt_dir).values()))
        print(_layout(_load(a.ckpt_dir, dirs[0]), len(dirs)))
    else:
        convert(a.src, a.dst, a.cmd)


if __name__ == "__main__":
    main()
