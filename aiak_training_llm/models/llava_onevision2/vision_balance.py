"""Balance vision-tower work across data-parallel ranks (--vision-dp-balance).

Why
---
Every DP rank encodes the images of its own packed bin, and bins differ a lot in how
many image/frame patches they carry (a bin of long-video frames can have tens of times
the patches of a bin with a few images). The step waits for the slowest rank, so the
vision tower's cost is set by the heaviest bin of the step, not the average.

What
----
Images (grid_thw rows: one image or one frame) are independent units for the vision tower
(RoPE and attention are per image; with vision_frame_windows_size=1 every frame is its own
attention window) and for the adapter (per token). So each microbatch:

1. DP ranks all-gather their image grids (a few ints per image).
2. Every rank computes the same deterministic assignment: each rank keeps its own images up
   to the balanced load (average patches per rank, largest first), and the surplus images are
   given to the least-loaded ranks (largest first).
3. Patches (and patch_positions) are sent to the computing rank (all_to_all, no grad needed).
4. Each rank runs vision tower + adapter on the images it computes.
5. The embeddings are sent back to the owning rank (differentiable all_to_all) and put back
   in the original image order, so the caller gets exactly what local encoding would return.

Backward goes through the same all_to_all in reverse, so every image's vision gradient is
computed on the rank that ran its forward; the DP gradient all-reduce then sums all images'
contributions, as before. The math is unchanged; only which GPU encodes which image moves.

Constraints: all ranks of the DP group must call balance() for every microbatch (also when a
rank has no images), and the DP group must not have asynchronous gradient reductions in
flight during backward (--overlap-grad-reduce), which could interleave with the all_to_all.
"""

import os
from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_to_all_single as all_to_all_single_autograd


def _gather_grids(grid_thw: torch.Tensor, flags: torch.Tensor, group) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """All-gather each rank's [n_i, 3] grid (variable n_i) and a small per-rank flag vector."""
    world = dist.get_world_size(group)
    dev = grid_thw.device
    n = torch.tensor([grid_thw.shape[0]], dtype=torch.int64, device=dev)
    ns = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(ns, n, group=group)
    ns = [int(x.item()) for x in ns]
    nmax = max(ns)
    padded = torch.zeros((nmax, 3), dtype=torch.int64, device=dev)
    padded[: grid_thw.shape[0]] = grid_thw.to(torch.int64)
    bufs = [torch.zeros_like(padded) for _ in range(world)]
    dist.all_gather(bufs, padded, group=group)
    fl = [torch.zeros_like(flags) for _ in range(world)]
    dist.all_gather(fl, flags, group=group)
    return [b[:k].cpu() for b, k in zip(bufs, ns)], torch.stack(fl).cpu()


_calls = 0


def _log_loads(sizes, compute, group):
    """VISION_BALANCE_LOG=N: every N calls, group rank 0 prints max/mean patches before and after."""
    global _calls
    _calls += 1
    every = int(os.environ.get("VISION_BALANCE_LOG", "0"))
    if every <= 0 or _calls % every or dist.get_rank(group) != 0:
        return
    world = len(sizes)
    before = [sum(s) for s in sizes]
    after = [0] * world
    for r, s in enumerate(sizes):
        for i, n in enumerate(s):
            after[compute[r][i]] += n
    mean = sum(before) / world
    moved = sum(n for r, s in enumerate(sizes) for i, n in enumerate(s) if compute[r][i] != r)
    print(f"[VISION_BALANCE] call {_calls}: patches max {max(before)} -> {max(after)} (mean {mean:.0f}), "
          f"moved {moved} ({100 * moved / max(1, sum(before)):.1f}%)", flush=True)


def plan_assignment(sizes: List[List[int]]) -> List[List[int]]:
    """sizes[r][i] = patches of image i on rank r -> compute_rank[r][i] (deterministic)."""
    world = len(sizes)
    total = sum(sum(s) for s in sizes)
    target = total / world
    compute = [[-1] * len(s) for s in sizes]
    load = [0] * world
    surplus = []
    for r, s in enumerate(sizes):
        for i in sorted(range(len(s)), key=lambda i: (-s[i], i)):  # keep largest local images first
            if load[r] + s[i] <= target or load[r] == 0:
                compute[r][i] = r
                load[r] += s[i]
            else:
                surplus.append((s[i], r, i))
    for size, r, i in sorted(surplus, key=lambda t: (-t[0], t[1], t[2])):
        d = min(range(world), key=lambda k: (load[k], k))
        compute[r][i] = d
        load[d] += size
    return compute


def balance(
    images: torch.Tensor,
    grid_thw: torch.Tensor,
    patch_positions: Optional[torch.Tensor],
    encode: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    merge: int,
    out_dim: int,
    group,
) -> torch.Tensor:
    """Encode this rank's images with the vision work balanced across `group`.

    images: [n_patches, C] (may be empty), grid_thw: [n_images, 3], patch_positions: [n_patches, 3]
    or None. encode(images, grid_thw, patch_positions) -> [n_patches // merge, out_dim].
    Returns [n_patches // merge, out_dim] in the original image order.
    """
    world = dist.get_world_size(group)
    me = dist.get_rank(group)
    dev = images.device
    sizes_local = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.int64) if grid_thw.numel() else \
        torch.zeros((0,), dtype=torch.int64, device=dev)
    consistent = int(sizes_local.sum().item()) == images.shape[0] and (
        patch_positions is None or patch_positions.shape[0] == images.shape[0])
    flags = torch.tensor([int(patch_positions is not None), int(consistent)], dtype=torch.int64, device=dev)
    grids, fl = _gather_grids(grid_thw.reshape(-1, 3), flags, group)
    has_pp = fl[:, 0].tolist()
    # All ranks must take the same branch: fall back to local encoding everywhere unless every
    # rank's inputs are consistent and patch_positions presence agrees.
    if not all(fl[:, 1].tolist()) or (any(has_pp) and not all(has_pp)):
        return encode(images, grid_thw, patch_positions) if images.shape[0] else images.new_zeros((0, out_dim))

    sizes = [(g[:, 0] * g[:, 1] * g[:, 2]).tolist() for g in grids]
    compute = plan_assignment(sizes)
    use_pp = bool(has_pp[0])
    _log_loads(sizes, compute, group)

    # ---- dispatch patches: for each dest d, my images computed on d (ascending index) ----
    starts = [0]
    for s in sizes[me]:
        starts.append(starts[-1] + s)
    send_idx = [[i for i in range(len(sizes[me])) if compute[me][i] == d] for d in range(world)]
    rows = [torch.arange(starts[i], starts[i + 1], device=dev) for d in range(world) for i in send_idx[d]]
    row_index = torch.cat(rows) if rows else torch.zeros((0,), dtype=torch.long, device=dev)
    in_split = [sum(sizes[me][i] for i in send_idx[d]) for d in range(world)]
    # what I compute: for each src s, its images assigned to me (ascending index)
    recv_units = [(s, i) for s in range(world) for i in range(len(sizes[s])) if compute[s][i] == me]
    out_split = [sum(sizes[s][i] for (s2, i) in recv_units if s2 == s) for s in range(world)]

    def a2a(t):
        src = t.index_select(0, row_index).contiguous()
        out = t.new_empty((sum(out_split),) + tuple(t.shape[1:]))
        dist.all_to_all_single(out, src, out_split, in_split, group=group)
        return out

    with torch.no_grad():
        imgs_c = a2a(images)
        pp_c = a2a(patch_positions) if use_pp else None
    grid_c = torch.stack([grids[s][i] for (s, i) in recv_units]).to(dev) if recv_units else \
        torch.zeros((0, 3), dtype=torch.int64, device=dev)

    # ---- encode what I compute ----
    if recv_units:
        emb_c = encode(imgs_c, grid_c, pp_c)
    else:
        # Must still require grad: every rank has to reach the backward all_to_all.
        emb_c = images.new_zeros((0, out_dim), requires_grad=True)

    # ---- return embeddings to owners (differentiable) ----
    send_tok = [out_split[s] // merge for s in range(world)]        # emb_c is grouped by src rank
    recv_tok = [in_split[d] // merge for d in range(world)]          # my images, grouped by computing rank
    emb_back = torch.empty((sum(recv_tok), out_dim), dtype=emb_c.dtype, device=dev)
    emb_back = all_to_all_single_autograd(emb_back, emb_c.contiguous(), recv_tok, send_tok, group=group)

    # emb_back rows: for d in ranks, my images send_idx[d] in ascending order -> restore image order
    tok_starts = [0]
    for s in sizes[me]:
        tok_starts.append(tok_starts[-1] + s // merge)
    order = [i for d in range(world) for i in send_idx[d]]
    pos = torch.empty((tok_starts[-1],), dtype=torch.long, device=dev)
    cur = 0
    for i in order:
        n = sizes[me][i] // merge
        pos[tok_starts[i]:tok_starts[i] + n] = torch.arange(cur, cur + n, device=dev)
        cur += n
    return emb_back.index_select(0, pos)
