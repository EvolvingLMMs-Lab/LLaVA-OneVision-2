"""Cross-rank load balancing of an iteration's packed sequences (switch OV2_OPT_BALANCE).

An iteration trains on the global batch's packed sequences, which the data-parallel ranks' loaders yield (micro-batches
per rank = global batch / data-parallel size, one packed sequence each). Their compute cost differs a lot (the vision
content of a packed sequence: from no image to ~32k patches), and every rank waits for the slowest one at the gradient
reduce-scatter. This iterator wraps a rank's training data iterator: at the first next() of an iteration it takes the
rank's own sequences from the loader (as before), exchanges a cost estimate and a description of every sequence with
the other ranks (gloo, CPU only), assigns the iteration's sequences to the ranks, the same number each, minimising the
largest estimated load (the same deterministic computation on every rank, keeping as many sequences as possible on
the rank that loaded them), and moves the reassigned sequences' tensors between the GPUs (NCCL point-to-point on a
dedicated communicator and stream, then a copy into pinned host memory). next() then yields, one per call, the rank's
assigned sequences (the locally loaded ones first), each one exactly the packed-sequence dict the loader produced (same
keys, same non-tensor values, every tensor with the same dtype, shape and values, on the host).

What does not change: which sequences form the iteration, every tensor of every sequence, the number of sequences per
rank and iteration, and that each sequence is trained exactly once with its own token-mean (the loss and Megatron's
schedule are untouched: each sequence is still one micro-batch, normalised by its own loss-token count and by the number
of micro-batches, and the gradient reduction averages over the ranks). The objective is therefore the same; only the
order in which per-sequence gradients are summed changes.

startup_check() runs first and keeps the default data path, with a warning, where balancing is not supported or cannot
help: TP, PP or CP above 1 (not validated, and process groups below assume one data-parallel group), one data-parallel
rank, one sequence per rank and iteration, or a group that spans nodes (moved sequences would cross the node fabric;
OV2_OPT_BALANCE_CROSS_NODE=1 allows it).
"""
import collections
import os

import torch

from megatron.core import parallel_state as mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches

IMAGE_TOKEN_ID = 151655
FRAME_WINDOW = 4                 # OneVision encoder frame_windows_size (frames per attention window)

# Per-sequence cost model (ms), calibrated on a Blackwell GPU for the 4B quick-start model with the Blackwell preset's
# kernels: per-micro-batch kernel time of a profiled run (R^2 0.9985), and per-rank GPU timelines of a balanced run
# (sequences with little vision work are bound by the host's kernel launches, ~275 ms per sequence).
_C0 = 204.05 / 8190            # per LM token (LM layers, loss, fixed per-sequence work)
_CP = 3.4492e-3                # per vision patch (vision GEMMs, adapter, memory-bound work)
_CW = 0.3459e-6                # per window-length^2 (vision attention)
_CL = 0.8542e-6                # per sub-sample-length^2 (causal LM attention)
_HOST_FLOOR = 275.0            # host launch time of one sequence's forward + backward
_RC = None                     # recompute costs of the OV2_OPT_MEM_BUDGET_GIB planner (ms), if given
if os.environ.get('OV2_OPT_BALANCE_COSTS'):
    # OV2_OPT_BALANCE_COSTS=<ms per 8190 tokens>,<ms per patch>,<ms per window-length^2>,<ms per sub-sample-length^2>,
    # <host floor ms>[,<act LM ms per 8190 tokens>,<act vision ms per patch>,<fc1 vision ms per patch>,<fc1 LM ms per
    # layer and 8190 tokens>]: the same cost model calibrated on another GPU (on Hopper the GEMM / attention /
    # host-launch balance differs from Blackwell's), plus, with the per-micro-batch memory planner, the cost of the
    # recomputation it plans for the sequence. Only the assignment of the iteration's sequences to ranks changes.
    _c = [float(x) for x in os.environ['OV2_OPT_BALANCE_COSTS'].split(',')]
    if len(_c) not in (5, 9):
        raise ValueError('OV2_OPT_BALANCE_COSTS: expected 5 or 9 comma-separated values')
    _C0, _CP, _CW, _CL, _HOST_FLOOR = _c[0] / 8190, _c[1], _c[2], _c[3], _c[4]
    if len(_c) == 9:
        _RC = _c[5:]


_ACTIVE = None


def prepull():
    """OV2_OPT_BALANCE_PREPULL: called right before the gradient reduce-scatter (the host has launched the iteration's
    last backward and next waits for the GPU): take the next iteration's own sequences from the loader and describe
    them now (CPU only), so the refill at the next iteration's first next() only exchanges and assigns. Not done when a
    checkpoint or the end of training follows this iteration (the loader state saved then stays exact)."""
    it = _ACTIVE
    if it is None or not it._prepull_on or it._prepulled is not None or it._queue:
        return
    try:
        from megatron.training import get_args
        a = get_args()
        nxt = int(getattr(a, 'curr_iteration', 0)) + 1
        if a.save and a.save_interval and nxt % a.save_interval == 0:
            return
        if nxt >= int(a.train_iters) or (a.exit_interval and nxt % a.exit_interval == 0):
            return
    except Exception:
        return
    n = get_num_microbatches()
    own = [next(it._inner) for _ in range(n)]
    it._prepulled = (own, [(sequence_cost(d), _spec(d)) for d in own])


def sequence_cost(d):
    """Estimated training time (fwd + bwd, ms) of one packed sequence, from its host tensors."""
    tokens = d.get('tokens')
    n_tok = int(tokens.shape[-1]) if isinstance(tokens, torch.Tensor) else 0
    cost = _C0 * n_tok
    cu = d.get('cu_lengths')
    if isinstance(cu, torch.Tensor) and cu.numel() > 1:
        b = cu.reshape(-1).tolist()
        cost += _CL * sum((y - x) * (y - x) for x, y in zip(b[:-1], b[1:]))
    else:
        cost += _CL * n_tok * n_tok
    thw = d.get('image_grid_thw')
    has_image = isinstance(tokens, torch.Tensor) and bool((tokens == IMAGE_TOKEN_ID).any())
    if has_image and isinstance(thw, torch.Tensor) and thw.numel():
        nv = 0
        sq = 0
        for t, h, w in thw.reshape(-1, 3).tolist():
            hw = h * w
            if t > FRAME_WINDOW:
                full, rem = divmod(t, FRAME_WINDOW)
                sq += full * (FRAME_WINDOW * hw) ** 2 + (rem * hw) ** 2
            else:
                sq += (t * hw) ** 2
            nv += t * hw
        cost += _CW * sq + _CP * nv
    if _RC is not None:
        from aiak_training_llm.models.custom.common import activation_memory
        if activation_memory.budget_enabled():
            nv_all = 0
            for key in ('image_grid_thw', 'video_grid_thw'):
                g = d.get(key)
                if isinstance(g, torch.Tensor) and g.numel():
                    nv_all += sum(t * h * w for t, h, w in g.reshape(-1, 3).tolist())
            p = activation_memory.plan(n_tok / 1000.0, nv_all / 1000.0)
            cost += (_RC[0] * n_tok / 8190 * p['act_lm'] + _RC[1] * nv_all * p['act_vision']
                     + _RC[2] * nv_all * p['fc1_vision'] + _RC[3] * p['fc1_lm'] * n_tok / 8190)
    return max(cost, _HOST_FLOOR)


def startup_problem(dp_size, job_size, micro_batches, nodes, cross_node):
    """Why OV2_OPT_BALANCE stays off for this job, or None if it can run."""
    if dp_size != job_size:
        return f'the data-parallel group holds {dp_size} of the {job_size} ranks; only TP, PP and CP 1 are supported'
    if dp_size < 2:
        return 'there is only one data-parallel rank'
    if micro_batches < 2:
        return (f'{micro_batches} sequence per rank and iteration leaves nothing to balance; it needs a global batch '
                'size of at least 2 x data-parallel size x micro-batch size')
    if nodes > 1 and not cross_node:
        return (f'the data-parallel group spans {nodes} nodes and moved sequences, pixels included, would cross the '
                'node fabric; set OV2_OPT_BALANCE_CROSS_NODE=1 to balance anyway')
    return None


def startup_check():
    """True if BalancedDataIterator should wrap the data iterator. Its constructor creates process groups, which every
    rank must enter together, so this runs first and gives the same answer on every rank."""
    from megatron.training import get_args
    args = get_args()
    group = mpu.get_data_parallel_group()
    dp_size = torch.distributed.get_world_size(group)
    job_size = torch.distributed.get_world_size()
    ranks_per_node = int(os.environ.get('LOCAL_WORLD_SIZE', 0)) or torch.cuda.device_count()
    nodes = len({r // ranks_per_node for r in torch.distributed.get_process_group_ranks(group)})
    problem = startup_problem(dp_size, job_size, args.global_batch_size // (args.micro_batch_size * dp_size), nodes,
                              os.environ.get('OV2_OPT_BALANCE_CROSS_NODE', '0') == '1')
    if dp_size == job_size:
        # every rank is here (with a smaller group some ranks have no data iterator, and all ranks stop without a
        # collective); ranks can disagree when the launcher does not pass the same environment to every node
        agreed = torch.tensor([problem is None], dtype=torch.int32, device=torch.cuda.current_device())
        torch.distributed.all_reduce(agreed, op=torch.distributed.ReduceOp.MIN)
        if problem is None and not agreed.item():
            problem = 'another rank found a problem (is the environment the same on every node?)'
    if problem is not None and torch.distributed.get_rank() == 0:
        print(f'[ov2-balance] WARNING: OV2_OPT_BALANCE is off for this run: {problem}', flush=True)
    return problem is None


def match_groups(groups, world):
    """perm[g] = the rank of group g, keeping the most sequences on the rank that loaded them: a maximum-weight
    bipartite matching (Hungarian algorithm on integers, deterministic, about 0.5 ms at world 48)."""
    cost = [[0] * world for _ in range(world)]
    for g, group in enumerate(groups):
        for c in group:
            cost[g][c[1]] -= 1
    inf = float('inf')
    u, v = [0] * (world + 1), [0] * (world + 1)
    owner, way = [0] * (world + 1), [0] * (world + 1)      # owner[r + 1]: 1-based group matched to rank r
    for g in range(1, world + 1):
        owner[0], col = g, 0
        slack, used = [inf] * (world + 1), [False] * (world + 1)
        while True:
            used[col] = True
            row, delta, nxt = owner[col], inf, 0
            for j in range(1, world + 1):
                if not used[j]:
                    cur = cost[row - 1][j - 1] - u[row] - v[j]
                    if cur < slack[j]:
                        slack[j], way[j] = cur, col
                    if slack[j] < delta:
                        delta, nxt = slack[j], j
            for j in range(world + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    slack[j] -= delta
            col = nxt
            if owner[col] == 0:
                break
        while col:
            owner[col] = owner[way[col]]
            col = way[col]
    perm = [0] * world
    for j in range(1, world + 1):
        perm[owner[j] - 1] = j - 1
    return tuple(perm)


def assign(costs, world, per_rank):
    """costs: list of (cost, src_rank, index), world * per_rank entries. Returns for every rank the list of (src_rank,
    index) it trains, per_rank each: LPT then pairwise-swap refinement of the largest load, then the group -> rank
    matching that keeps the most sequences where they were loaded. Deterministic (ties broken by (src_rank, index))."""
    order = sorted(costs, key=lambda x: (-x[0], x[1], x[2]))
    groups = [[] for _ in range(world)]
    loads = [0.0] * world
    for c in order:
        g = min((g for g in range(world) if len(groups[g]) < per_rank), key=lambda g: (loads[g], g))
        groups[g].append(c)
        loads[g] += c[0]
    for _ in range(64):
        hi = max(range(world), key=lambda g: (loads[g], -g))
        best = None
        for lo in range(world):
            if lo == hi:
                continue
            for a, ca in enumerate(groups[hi]):
                for b, cb in enumerate(groups[lo]):
                    d = ca[0] - cb[0]
                    if d <= 0:
                        continue
                    new_max = max(loads[hi] - d, loads[lo] + d)
                    if new_max < loads[hi] - 1e-9 and (best is None or new_max < best[0]):
                        best = (new_max, lo, a, b, d)
        if best is None:
            break
        _, lo, a, b, d = best
        groups[hi][a], groups[lo][b] = groups[lo][b], groups[hi][a]
        loads[hi] -= d
        loads[lo] += d
    perm = match_groups(groups, world)
    out = [None] * world
    for g in range(world):
        r = perm[g]
        mine = sorted(groups[g], key=lambda c: (c[1] != r, c[1], c[2]))   # locally loaded first
        out[r] = [(c[1], c[2]) for c in mine]
    return out, [loads[g] for g in range(world)], perm


def _spec(d):
    """Description of a packed-sequence dict: per key, ('T', dtype, shape) for a tensor, ('V', value) otherwise."""
    out = []
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            out.append((k, 'T', v.dtype, tuple(v.shape)))
        else:
            out.append((k, 'V', v))
    return out


def _wire(t):
    """A contiguous tensor NCCL can move (bool as uint8)."""
    t = t.contiguous()
    return t.view(torch.uint8) if t.dtype == torch.bool else t


class BalancedDataIterator:
    """Wraps the training data iterator; see the module docstring."""

    def __init__(self, inner):
        self._inner = inner
        self._queue = collections.deque()
        group = mpu.get_data_parallel_group()
        self._world = torch.distributed.get_world_size(group)
        self._rank = torch.distributed.get_rank(group)
        ranks = [torch.distributed.get_global_rank(group, r) for r in range(self._world)]
        self._granks = ranks
        # metadata on the CPU (no wait for the GPU work still queued from the previous iteration), tensors on a
        # communicator of their own (its stream does not queue behind the data-parallel collectives)
        self._gloo = torch.distributed.new_group(ranks, backend='gloo')
        self._nccl = torch.distributed.new_group(ranks, backend='nccl')
        self._stream = torch.cuda.Stream()
        self._device = torch.device('cuda', torch.cuda.current_device())
        # the first operation on the new communicator involves every rank (later batched point-to-point calls
        # may then involve only the ranks that exchange something)
        warm = torch.ones(1, device=self._device)
        torch.distributed.all_reduce(warm, group=self._nccl)
        torch.cuda.current_stream().synchronize()
        self._refills = 0
        self._moved = 0
        self._est_max = 0.0
        self._est_now = 0.0
        self._verbose = os.environ.get('OV2_OPT_BALANCE_LOG', '1') == '1'
        self._prev = None
        self._prepull_on = os.environ.get('OV2_OPT_BALANCE_PREPULL', '0') == '1'
        self._prepulled = None
        global _ACTIVE
        _ACTIVE = self

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._queue:
            self._refill()
        item = self._queue.popleft()
        return item() if callable(item) else item

    def _refill(self):
        prev, self._prev = self._prev, None
        if prev is not None:
            # the previous iteration's transfers (long finished): release their GPU buffers
            prev['done'].synchronize()
        n = get_num_microbatches()
        pre, self._prepulled = self._prepulled, None
        if pre is not None and len(pre[0]) == n:
            own, metas = pre
        else:
            own = [next(self._inner) for _ in range(n)]
            metas = [(sequence_cost(d), _spec(d)) for d in own]
        every = [None] * self._world
        torch.distributed.all_gather_object(every, metas, group=self._gloo)
        costs = [(every[r][j][0], r, j) for r in range(self._world) for j in range(n)]
        plan, loads, _ = assign(costs, self._world, n)
        me = self._rank
        mine = plan[me]
        sends, recvs = [], []
        for dst in range(self._world):
            if dst == me:
                continue
            for r, j in plan[dst]:
                if r == me:
                    sends.append((dst, j))
        for r, j in mine:
            if r != me:
                recvs.append((r, j))
        pending = {}
        if sends or recvs:
            ops, keep = [], []
            with torch.cuda.stream(self._stream):
                for dst, j in sends:
                    for k, kind, *rest in _spec(own[j]):
                        if kind != 'T' or own[j][k].numel() == 0:
                            continue
                        g = _wire(own[j][k]).to(self._device, non_blocking=True)
                        keep.append(g)
                        ops.append(torch.distributed.P2POp(torch.distributed.isend, g, self._granks[dst], self._nccl))
                bufs = {}
                for r, j in recvs:
                    b = {}
                    for k, kind, *rest in every[r][j][1]:
                        if kind != 'T':
                            continue
                        dtype, shape = rest
                        wdtype = torch.uint8 if dtype == torch.bool else dtype
                        g = torch.empty(shape, dtype=wdtype, device=self._device)
                        b[k] = g
                        if g.numel():
                            ops.append(torch.distributed.P2POp(torch.distributed.irecv, g, self._granks[r],
                                                               self._nccl))
                    bufs[(r, j)] = b
                works = torch.distributed.batch_isend_irecv(ops) if ops else []
                for w in works:
                    w.wait()
                host = {}
                for (r, j), b in bufs.items():
                    h = {}
                    for k, g in b.items():
                        c = torch.empty(g.shape, dtype=g.dtype, pin_memory=True)
                        c.copy_(g, non_blocking=True)
                        h[k] = c
                    host[(r, j)] = h
                done = torch.cuda.Event()
                done.record(self._stream)
            state = {'keep': keep, 'bufs': bufs, 'done': done, 'synced': False}

            for r, j in recvs:
                def make(r=r, j=j, h=host[(r, j)], spec=every[r][j][1], st=state):
                    if not st['synced']:
                        st['done'].synchronize()
                        st['keep'].clear()
                        st['bufs'].clear()
                        st['synced'] = True
                    d = {}
                    for k, kind, *rest in spec:
                        if kind == 'T':
                            dtype, shape = rest
                            t = h.get(k)
                            if t is None:                      # empty tensor
                                t = torch.empty(shape, dtype=dtype)
                            elif dtype == torch.bool:
                                t = t.view(torch.bool)
                            d[k] = t
                        else:
                            d[k] = rest[0]
                    return d
                pending[(r, j)] = make
            self._prev = state
        for r, j in mine:
            self._queue.append(own[j] if r == me else pending[(r, j)])
        self._refills += 1
        self._moved += len(recvs)
        before = [sum(every[r][j][0] for j in range(n)) for r in range(self._world)]
        self._est_now += max(before)
        self._est_max += max(loads)
        if self._verbose and me == 0 and (self._refills in (1, 2) or self._refills % 25 == 0):
            print(f'[ov2-balance] refill {self._refills}: estimated max load {max(before):.0f} -> {max(loads):.0f} '
                  f'(mean {sum(before) / len(before):.0f}); moved to rank 0: {len(recvs)}; average over refills: '
                  f'{self._est_now / self._refills:.0f} -> {self._est_max / self._refills:.0f}', flush=True)
