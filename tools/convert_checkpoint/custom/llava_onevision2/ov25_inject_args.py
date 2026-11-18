#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Copy an mcore checkpoint (mp_rank_*) into a working dir, injecting a megatron
``args`` namespace into every shard.

Why: the Qwen3.5 forward language converter (ov25_hf_to_mcore.py) writes a
minimal ``{"model": ...}`` with NO megatron ``args`` (it does not need it -- the
language reverse converter reconstructs everything from CLI + param shapes). But
the *vision-tower* mcore->hf step reuses the generic ``model.py``, whose
``McoreConfig.load`` requires ``args`` in rank-0. Real Megatron *training*
checkpoints always carry ``args``; only the converter-produced checkpoint lacks
it. This helper makes such a checkpoint usable by the unchanged vision converter,
WITHOUT mutating the original checkpoint (writes to a separate working dir) and
WITHOUT touching the forward converters.

``args`` is architecture/parallelism metadata only (no weights); it is taken from
a throwaway forward vision conversion of the reference HF model (identical arch),
so it is exactly what a training checkpoint would carry.
"""
import argparse
import os
import shutil

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--args_from", required=True,
                    help="mcore release dir (mp_rank_*) that HAS args (e.g. forward vision output)")
    ap.add_argument("--src", required=True, help="source mcore release dir (args-less)")
    ap.add_argument("--dst", required=True, help="destination working release dir (args injected)")
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()

    src_args_path = os.path.join(a.args_from, "mp_rank_00", "model_optim_rng.pt")
    meta_src = torch.load(src_args_path, map_location="cpu", weights_only=False)
    # copy all non-'model' metadata keys the vision reverse (model.py) expects
    # (args, checkpoint_version, iteration, ...).
    meta = {k: v for k, v in meta_src.items() if k != "model"}
    assert "args" in meta, f"no 'args' in {src_args_path}"

    os.makedirs(a.dst, exist_ok=True)
    for t in range(a.tp):
        sub = f"mp_rank_{t:02d}"
        src_p = os.path.join(a.src, sub, "model_optim_rng.pt")
        dst_dir = os.path.join(a.dst, sub)
        os.makedirs(dst_dir, exist_ok=True)
        ck = torch.load(src_p, map_location="cpu", weights_only=False)
        for k, v in meta.items():
            ck.setdefault(k, v)
        torch.save(ck, os.path.join(dst_dir, "model_optim_rng.pt"))
        print(f"[inject_args] rank {t}: {src_p} + {sorted(meta)} -> {dst_dir}")


if __name__ == "__main__":
    main()
