# Best-speed 16-node training (SP + TP=4)

Single, self-contained training script (node list + all args inline, runs torchrun directly):

    date0624_ax_stage_1p5_recap_w47m_packed_64k_16n_tp4_gbs32_2ep_SP_from_date0608.sh

## Config baked in
- 16 nodes (list_ip inline: 172.16.5.32-36,38-47,49), TP=4 PP=1 -> DP=32
- --sequence-parallel ON (SP tail-fix in baseline get_batch), recompute full/uniform/1
- seq-length 102400 (recap bins exceed 65536; do NOT lower), MBS=1, GBS=32 (2ep)
- DATA: recap_w47m yaml ; CHECKPOINT: resharded TP=4 (date0608-release-load-TP4)
- save-interval 400 ; code = /vlm/yinxie/code/OV21/baseline (fixes included)

## Run it
Run on each of the 16 nodes inside the container (same way date0622 is run); the script
auto-detects NODE_RANK from the node's IP in list_ip and launches torchrun for that rank:

    bash date0624_ax_stage_1p5_recap_w47m_packed_64k_16n_tp4_gbs32_2ep_SP_from_date0608.sh

(Defaults: TP=4 PP=1 SEQ_LEN=102400 MBS=1 GBS=32, NSTEP auto from EPOCHS=2.)
GBS must stay a multiple of 32 (DP=32). 48 is invalid at 16 nodes.

## Speed (measured on A800)
SP+TP=4 is ~1.6x the old NoSP/TP=8 baseline. ~2 days/epoch on 16 nodes (vs ~15 days on 2).
