# Best-speed training: SP + TP=4 (LLaVA-OneVision-2-4B, 64k packed)

`date0624_BEST_sp_tp4_recap_64k.sh` is the fastest verified config: **~1.6x** the
throughput of the old NoSP/TP=8 production run, numerically identical loss.

| config | ms/iter (2-node, GBS=4, 64k, same bins) | speedup |
|--------|------------------------------------------|---------|
| NoSP TP=8 (old prod) | 25,209 | 1.00x |
| SP TP=8              | 20,554 | 1.23x |
| **SP TP=4 (this)**   | **15,862** | **1.59x** |

## What changed vs the old prod script
1. `--sequence-parallel` is ON (the iter-1 NaN it used to cause is fixed in baseline).
2. `TP=4` instead of 8 (SP frees the memory; TP=8 was over-sharded for a 4B model).
3. Code path = the FIXED baseline (`AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2.5`).
4. Data = recap yaml with weights corrected to real bin counts.
5. `--tp-comm-overlap` is intentionally NOT used (it is slower on this hardware).

## ONE-TIME PREREQUISITE: reshard the load checkpoint TP=8 -> TP=4
The pretrained release ckpt is TP=8-sharded (torch format) and CANNOT auto-reshard
into a TP=4 run. Produce a TP=4 copy once, then point CHECKPOINT_PATH at it:

  - Conversion tooling lives in: examples/llava_onevision2/convert/
  - Target path expected by the script (override with --env CHECKPOINT_PATH=... if different):
      /workspace/LLaVA-OneVision-2.5/checkpoints/date0608-release-load-TP4
  - If you don't have a TP-reshard converter handy, ping me and I'll produce it
    (load TP=8 -> save TP=4), or run the original release-load step at --tp 4.

Until the TP=4 ckpt exists, the run will fail at load with a shape mismatch.

## Launch (16 nodes, real training)
Uses the generic launcher (sshes into each node and `docker exec`s the script):

```bash
bash examples/llava_onevision2/quick_start_4b/launch_date05_training.sh \
  --script /workspace/LLaVA-OneVision-2.5/examples/llava_onevision2/quick_start_4b/date0624_BEST_sp_tp4_recap_64k.sh \
  --ips "192.0.2.19 192.0.2.20 ... <all 16 node IPs in rank order>" \
  --tp 4 --pp 1 --seq-len 102400 --mbs 1 --gbs 32 \
  --container llava_megatron_container_ax \
  --env AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2.5
```

Notes:
- `--gbs 32` @ 16 nodes => DP=32 => 1 microbatch/rank. GBS must stay divisible by 32.
- `--seq-len 102400`: required — recap bins exceed 65536; do NOT lower it.
- Override data/ckpt/output if needed via `--env DATA_PATH=...`, `--env CHECKPOINT_PATH=...`,
  `--env OUTPUT_DIR=...`.
- Monitor: `ssh <node> "docker exec llava_megatron_container_ax bash -lc 'tail -f /tmp/<launch-log-name>_launch.log'"`
  (the per-iter `iteration .../... | lm loss | ...` lines print on the report rank).
- Sanity on first iters: `number of nan iterations: 0`, lm loss finite, max-alloc ~50 GB/80.

## Quick single-/two-node smoke test (before the 16-node run)
```bash
bash examples/llava_onevision2/quick_start_4b/launch_date05_training.sh \
  --script /workspace/LLaVA-OneVision-2.5/examples/llava_onevision2/quick_start_4b/date0624_BEST_sp_tp4_recap_64k.sh \
  --ips "192.0.2.14 192.0.2.15" \
  --tp 4 --pp 1 --seq-len 102400 --mbs 1 --gbs 4 --nstep 205 \
  --env AIAK_TRAINING_PATH=/workspace/LLaVA-OneVision-2.5 \
  --env CHECKPOINT_PATH=<TP4 ckpt>
```
(2 nodes => DP=4 => GBS divisible by 4; expect ~15.8 s/iter, 0 NaN.)

## Hard-won facts baked into this config (do not re-litigate)
- SP NaN root cause = packing-pad tail vs cu_seqlens (fixed in baseline get_batch).
- tp-comm-overlap: ~16% slower here (no CUDA Multicast; needs UB_SKIPMC=1 to even run).
- num-workers: no effect at this scale (compute-bound).
- TP=2: OOMs on fp32 CE logits (vocab 152k / 2). TP=2+CP=2: CP+packed RoPE broken.
- The logs' `token/sec/GPU` field is unreliable; compare `elapsed time per iteration (ms)`.
