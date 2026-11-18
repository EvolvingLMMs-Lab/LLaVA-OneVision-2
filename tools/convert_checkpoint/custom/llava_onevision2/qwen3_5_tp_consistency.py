#!/usr/bin/env python3
"""Check / repair TP consistency of the replicated Qwen3.5 GDN parameters in an mcore checkpoint.

Why
---
GatedDeltaNet keeps its parameters REPLICATED on every tensor-parallel rank
(in_proj_qkvz, in_proj_ba, out_proj, out_norm, A_log, dt_bias, conv1d). Under sequence
parallelism each TP rank sees a different slice of the tokens, so these gradients must be
summed across the TP group. Before the SP-FIX in gated_deltanet.py they were not: each
rank trained its copy on its own half of every sequence and the copies drifted apart.

The mcore->HF converter reads the replicated tensors from TP rank 0 only. For a
checkpoint whose copies differ, that exports a rank-0-only model rather than the function
that was trained (training effectively ran the average of the per-rank copies), and
scores drop silently.

Usage
-----
  check  <ckpt_dir>             exit 1 if any replicated GDN tensor differs across TP ranks
  resync <src_dir> <dst_dir>    write a copy with those tensors replaced by the TP-rank mean

<ckpt_dir> is the directory that directly contains mp_rank_* (an iter_XXXXXXX dir or
release/). TP-sharded tensors and everything outside language_model are left untouched. For a
--gdn-head-parallel checkpoint only out_norm is replicated, so only it is checked/resynced.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

import torch

REPLICATED = re.compile(r"(in_proj_qkvz|in_proj_ba|out_proj|out_norm|A_log|dt_bias|conv1d)")
HEAD_PARALLEL_REPLICATED = re.compile(r"out_norm")
RANK_DIR = re.compile(r"^mp_rank_(\d{2})(?:_(\d{3}))?$")


def _shards_by_stage(ckpt_dir):
    """{pp_stage: [dir for tp_rank 0, 1, ...]} for mp_rank_TT or mp_rank_TT_PPP layouts."""
    stages = defaultdict(dict)
    for d in os.listdir(ckpt_dir):
        m = RANK_DIR.match(d)
        if m and os.path.isfile(os.path.join(ckpt_dir, d, "model_optim_rng.pt")):
            stages[m.group(2) or "000"][int(m.group(1))] = d
    if not stages:
        sys.exit(f"[tp_consistency] no mp_rank_*/model_optim_rng.pt under {ckpt_dir}")
    return {pp: [tps[t] for t in sorted(tps)] for pp, tps in sorted(stages.items())}


def _load(ckpt_dir, d):
    return torch.load(os.path.join(ckpt_dir, d, "model_optim_rng.pt"), map_location="cpu", weights_only=False)


def _replicated_pattern(ckpt):
    """With --gdn-head-parallel only out_norm stays replicated; the other GDN params are shards."""
    return HEAD_PARALLEL_REPLICATED if getattr(ckpt.get("args"), "gdn_head_parallel", False) else REPLICATED


def _replicated_keys(model, others, pattern=None):
    pattern = pattern or REPLICATED
    for k, v in model.items():
        if ("language_model" in k and pattern.search(k) and torch.is_tensor(v)
                and all(k in o and torch.is_tensor(o[k]) and o[k].shape == v.shape for o in others)):
            yield k


def check(ckpt_dir):
    stages = _shards_by_stage(ckpt_dir)
    if all(len(dirs) == 1 for dirs in stages.values()):
        print("[tp_consistency] TP=1 checkpoint, nothing to compare")
        return 0
    n_checked, bad = 0, []
    for pp, dirs in stages.items():
        cks = [_load(ckpt_dir, d) for d in dirs]
        models = [c["model"] for c in cks]
        for k in _replicated_keys(models[0], models[1:], _replicated_pattern(cks[0])):
            n_checked += 1
            for t, m in enumerate(models[1:], 1):
                if not torch.equal(models[0][k], m[k]):
                    diff = (models[0][k].float() - m[k].float()).abs().max().item()
                    bad.append(f"pp{pp} tp0 vs tp{t} {k} max|diff|={diff:.3e}")
    if n_checked == 0:
        print("[tp_consistency] WARNING: no replicated GDN tensors found (not a Qwen3.5 checkpoint?)")
        return 0
    if bad:
        print(f"[tp_consistency] FAILED: {len(bad)} replicated GDN tensor(s) differ across TP ranks:")
        for line in bad[:20]:
            print("   ", line)
        print("[tp_consistency] resync first: qwen3_5_tp_consistency.py resync <src> <dst>, then convert <dst>")
        return 1
    print(f"[tp_consistency] OK: {n_checked} replicated GDN tensors identical across TP ranks")
    return 0


def resync(src, dst):
    if os.path.abspath(src) == os.path.abspath(dst):
        sys.exit("[tp_consistency] resync: dst must differ from src (source is never modified)")
    stages = _shards_by_stage(src)
    for pp, dirs in stages.items():
        ckpts = [_load(src, d) for d in dirs]
        models = [c["model"] for c in ckpts]
        n = 0
        if len(models) > 1:
            for k in list(_replicated_keys(models[0], models[1:], _replicated_pattern(ckpts[0]))):
                mean = (sum(m[k].float() for m in models) / len(models)).to(models[0][k].dtype)
                for m in models:
                    m[k] = mean.clone()
                n += 1
        for d, c in zip(dirs, ckpts):
            os.makedirs(os.path.join(dst, d), exist_ok=True)
            torch.save(c, os.path.join(dst, d, "model_optim_rng.pt"))
        print(f"[tp_consistency] stage {pp}: averaged {n} replicated tensors over {len(models)} TP ranks")
    print(f"[tp_consistency] resynced checkpoint -> {dst}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").add_argument("ckpt_dir")
    r = sub.add_parser("resync")
    r.add_argument("src")
    r.add_argument("dst")
    a = ap.parse_args()
    sys.exit(check(a.ckpt_dir) if a.cmd == "check" else resync(a.src, a.dst))


if __name__ == "__main__":
    main()
