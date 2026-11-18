# Split a TP1/PP1 mcore release checkpoint (all routed experts in one file) into EP shards
# release/mp_rank_00_{ep:03d}/model_optim_rng.pt, the layout megatron's get_checkpoint_name
# expects when expert_model_parallel_size > 1 (TP1, PP1). Non-expert params are replicated;
# rank r keeps global experts [r*n, (r+1)*n) renumbered to local 0..n-1.
import argparse, os, re, torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="dir containing release/mp_rank_00/model_optim_rng.pt")
ap.add_argument("--dst", required=True)
ap.add_argument("--ep", type=int, required=True)
ap.add_argument("--num_experts", type=int, required=True)
a = ap.parse_args()

ck = torch.load(os.path.join(a.src, "release", "mp_rank_00", "model_optim_rng.pt"), map_location="cpu", weights_only=False)
model = ck["model"]
pat = re.compile(r"^(.*\.mlp\.experts\.linear_fc[12]\.weight)(\d+)$")
n_local = a.num_experts // a.ep
assert n_local * a.ep == a.num_experts
expert_keys = {k: pat.match(k) for k in model if pat.match(k)}
print(f"total keys {len(model)}, routed-expert keys {len(expert_keys)}")
for r in range(a.ep):
    lo, hi = r * n_local, (r + 1) * n_local
    m = {}
    for k, v in model.items():
        mt = expert_keys.get(k)
        if mt is None:
            m[k] = v
            continue
        g = int(mt.group(2))
        if lo <= g < hi:
            m[mt.group(1) + str(g - lo)] = v
    out = dict(ck)
    out["model"] = m
    d = os.path.join(a.dst, "release", f"mp_rank_00_{r:03d}")
    os.makedirs(d, exist_ok=True)
    torch.save(out, os.path.join(d, "model_optim_rng.pt"))
    n_e = sum(1 for k in m if pat.match(k))
    print(f"ep rank {r}: experts [{lo},{hi}) -> {n_e} expert tensors, {len(m)} keys")
open(os.path.join(a.dst, "latest_checkpointed_iteration.txt"), "w").write("release")
print("done")
