"""Activation-memory switches for GPUs that cannot hold all activations (80 GB GPUs; all off by default).

OV2_OPT_NO_RECOMPUTE keeps every activation: up to ~62 GiB per micro-batch on a packed sequence of 8192 tokens and 32k
vision patches, on top of ~33-40 GiB of weights, fp32 gradients and optimizer shards. These switches bring part of it
back by recomputation in the backward. None of them changes a value: the recomputed forward runs the same kernels on
the same inputs (bitwise identical in deterministic mode); only memory and time move.

  OV2_OPT_ACT_RECOMPUTE=all|lm|vision
        Output-discarding recomputation of the MLP activation (Megatron's dense_mlp_activation_func_recompute): the
        SwiGLU (language model) / bias + GELU (vision encoder) output is freed once fc2 has used it and recomputed from
        fc1's output right before fc2's backward.
  OV2_OPT_FC1_RECOMPUTE_VISION=<n>|all, OV2_OPT_FC1_RECOMPUTE_LM=<n>|all
        The same, extended to fc1 (with its fused pre-MLP layernorm) for the last n layers of that sub-model (after
        the layers of OV2_OPT_RECOMPUTE_*): only the MLP input is kept; layernorm + fc1 + activation are recomputed
        before fc2's backward (fc2 itself is not recomputed). Covers the activation too.
  OV2_OPT_RECOMPUTE_VISION=<n>|all, OV2_OPT_RECOMPUTE_LM=<n>|all
        Full recomputation (Megatron 'full' granularity, 'block' method) of the first n layers of that sub-model only;
        the other sub-model keeps its own setting (the command line or OV2_OPT_NO_RECOMPUTE).
  OV2_OPT_MEM_BUDGET_GIB=<g>
        A plan per micro-batch instead of a fixed set: before each training forward the planner estimates the
        micro-batch's activation memory from its LM tokens T and vision patches P (host-side shapes, no device read;
        linear model MEM_MODEL below) and turns on, cheapest per GiB first, only as much recomputation as needed to
        stay within g GiB: activation recompute (LM, vision), fc1 recompute of the vision MLPs, fc1 recompute of k
        language-model MLPs, then full recompute of vision / LM layers. Light sequences (few patches) run with little
        or no recomputation. The plan depends only on the sequence's shapes. Off, with a warning, with PP > 1.
  OV2_OPT_QKV_RECOMPUTE=1
        The language model's q / k / v are freed after the attention forward and rebuilt before its backward (needs
        OV2_OPT_FUSED_QKV_SPLIT and the FA4 / FA3 attention; fused_qk_norm_rope.QKVRecompute).
"""
import os

import torch

# layers per sub-model, from the transformer configs (configure())
_LAYERS = {}


def _count(name, total):
    v = os.environ.get(name, '').strip().lower()
    if not v or v == '0':
        return 0
    if v == 'all':
        return total
    n = int(v)
    if not 0 <= n <= total:
        raise ValueError(f'{name}={v}: expected 0..{total} or all')
    return n


def _sites(value):
    v = (value or '').strip().lower()
    if v in ('', '0'):
        return set()
    if v in ('1', 'all'):
        return {'lm', 'vision'}
    out = {x.strip() for x in v.split(',') if x.strip()}
    bad = out - {'lm', 'vision'}
    if bad:
        raise ValueError(f'OV2_OPT_ACT_RECOMPUTE: unknown {sorted(bad)} (all | lm | vision)')
    return out


def enabled():
    return any(os.environ.get(k, '').strip() not in ('', '0') for k in (
        'OV2_OPT_ACT_RECOMPUTE', 'OV2_OPT_FC1_RECOMPUTE_VISION', 'OV2_OPT_FC1_RECOMPUTE_LM', 'OV2_OPT_RECOMPUTE_VISION',
        'OV2_OPT_RECOMPUTE_LM', 'OV2_OPT_MEM_BUDGET_GIB'))


def configure(language_config, vision_config):
    """Per sub-model recomputation settings on the (already separate) transformer configs, before the model is built."""
    _LAYERS.update(lm=int(language_config.num_layers), vision=int(vision_config.num_layers))
    act = _sites(os.environ.get('OV2_OPT_ACT_RECOMPUTE'))
    for site, cfg in (('lm', language_config), ('vision', vision_config)):
        n = _count(f'OV2_OPT_RECOMPUTE_{site.upper()}', _LAYERS[site])
        if n:
            cfg.recompute_granularity = 'full'
            cfg.recompute_method = 'block'
            cfg.recompute_num_layers = n
            cfg.distribute_saved_activations = False
        if site in act:
            cfg.dense_mlp_activation_func_recompute = True


class _LeanRecompute:
    """Megatron's RecomputeManager (output-discarding recomputation) without the CPU / CUDA / tracker RNG state save
    and restore around it: the recomputed functions (layernorm, linear, activation) draw no random numbers (dropout is
    0 in this model), so the states are never touched; saves ~6 host-side state copies per call."""

    def __init__(self, run_function):
        self.run_function, self.outputs, self.ctx = run_function, None, None

    def checkpoint(self, *args):
        from megatron.core.tensor_parallel.random import ActivationCheckpoint
        out = ActivationCheckpoint.apply(self.run_function, self, *args)
        self.outputs = (out,)
        return out

    def discard_output(self):
        for o in self.outputs:
            o.untyped_storage().resize_(0)

    def recompute(self, _):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('Checkpointing is not compatible with .grad(), please use .backward() if possible')
        with torch.enable_grad():
            outputs = self.run_function(*self.ctx.saved_tensors)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        for o, r in zip(self.outputs, outputs):
            o.untyped_storage().resize_(r.untyped_storage().size())
            with torch.no_grad():
                o.untyped_storage().copy_(r.untyped_storage())
        self.ctx.outputs = outputs
        self.run_function = self.outputs = self.ctx = None


def _recompute_forward(self, hidden_states, with_fc1):
    """MLP.forward with the activation (and, with_fc1, the pre-MLP layernorm + fc1 before it) recomputed right before
    fc2's backward; the activation output is freed once fc2 has used it (fc2 itself is not recomputed)."""
    if with_fc1:
        def run(h):
            intermediate, bias = self.linear_fc1(h)
            return self.activation_func(intermediate, bias)
        manager = _LeanRecompute(run)
        intermediate = manager.checkpoint(hidden_states)
    else:
        intermediate, bias = self.linear_fc1(hidden_states)
        manager = _LeanRecompute(self.activation_func)
        intermediate = manager.checkpoint(intermediate, bias)
    output, output_bias = self.linear_fc2(intermediate)
    manager.discard_output()
    if output.requires_grad:
        output.register_hook(manager.recompute)
    return output, output_bias


def _fc1_recompute_forward(self, hidden_states):
    return _recompute_forward(self, hidden_states, True)


_PATCHED = False


def _patch_mlp():
    global _PATCHED
    if _PATCHED:
        return
    from megatron.core.transformer.mlp import MLP
    original = MLP.forward

    def forward(self, hidden_states):
        if self.training and torch.is_grad_enabled():
            if self.__dict__.get('_ov2_fc1_recompute', False):
                return _recompute_forward(self, hidden_states, True)
            if self.__dict__.get('_ov2_act_recompute', False):
                return _recompute_forward(self, hidden_states, False)
        return original(self, hidden_states)
    MLP.forward = forward
    _PATCHED = True


def mark_model(model):
    """After the model is built: flag the MLPs of the layers that get fc1 recomputation (patching MLP.forward once),
    or install the per micro-batch planner (OV2_OPT_MEM_BUDGET_GIB)."""
    if budget_enabled():
        from megatron.core import parallel_state
        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            return install_planner(model)
        # MEM_MODEL describes the whole model on one GPU, and a pipeline stage may not hold the vision encoder at all
        if torch.distributed.get_rank() == 0:
            print('[ov2-mem] WARNING: OV2_OPT_MEM_BUDGET_GIB is off for this run: pipeline parallelism is not supported. '
                  'With OV2_OPT_NO_RECOMPUTE, nothing limits activation memory.', flush=True)
    n_vis = _count('OV2_OPT_FC1_RECOMPUTE_VISION', _LAYERS['vision'])
    n_lm = _count('OV2_OPT_FC1_RECOMPUTE_LM', _LAYERS['lm'])
    if not (n_vis or n_lm):
        return model
    full = {'vision': _count('OV2_OPT_RECOMPUTE_VISION', _LAYERS['vision']),
            'lm': _count('OV2_OPT_RECOMPUTE_LM', _LAYERS['lm'])}
    from megatron.core.transformer.mlp import MLP
    import re
    marked = {'vision': 0, 'lm': 0}
    for name, m in model.named_modules():
        if not isinstance(m, MLP):
            continue
        mm = re.search(r'(vision_model|language_model)\.decoder\.layers\.(\d+)\.mlp$', name)
        if not mm:
            continue
        site = 'vision' if mm.group(1) == 'vision_model' else 'lm'
        idx, total, n = int(mm.group(2)), _LAYERS[site], (n_vis if site == 'vision' else n_lm)
        # the last n layers that are not fully recomputed
        if idx >= full[site] and idx >= total - n:
            m._ov2_fc1_recompute = True
            marked[site] += 1
    _patch_mlp()
    print(f'[ov2-mem] fc1 recomputation: {marked["vision"]} vision MLPs, {marked["lm"]} language-model MLPs', flush=True)
    return model


# ---------------------------------------------------------------- per micro-batch planner (OV2_OPT_MEM_BUDGET_GIB)

# Activation memory of one micro-batch (GiB; peak allocated during its forward + backward minus allocated at its
# forward start) of the 4B quick-start model with the presets' kernels (FA4, fused Triton kernels), with
# T = LM tokens / 1000 and P = vision patches / 1000, and what each recompute step saves; measured per micro-batch
# with torch.cuda memory statistics. Other models need their own coefficients.
MEM_MODEL = {
    'base': (-0.56, 4.397, 0.8142),      # a0, per 1k tokens, per 1k patches (no recompute; fit max |res| 0.01)
    'act_lm': 0.652,                     # per 1k tokens: SwiGLU output (36 layers)
    'act_vision': 0.183,                 # per 1k patches: bias + GELU output (24 layers)
    'fc1_vision': 0.229,                 # per 1k patches, beyond act_vision: pre-MLP layernorm + fc1 (+bias) outputs
    'fc1_lm_layer': 0.0410,              # per 1k tokens and LM layer, beyond act_lm: layernorm + fc1 outputs
    'full_vision_layer': 0.0145,         # per 1k patches and vision layer, beyond fc1/act: the rest of the layer
    'full_lm_layer': 0.0420,             # per 1k tokens and LM layer, beyond fc1/act: the rest of the layer
    'qkv_lm': 0.412,                     # per 1k tokens: q / k / v of the 36 LM attentions (OV2_OPT_QKV_RECOMPUTE=1)
}
_QKV = os.environ.get('OV2_OPT_QKV_RECOMPUTE', '0') == '1'
_BUDGET = float(os.environ.get('OV2_OPT_MEM_BUDGET_GIB', '0') or 0)
_STATE = {}


def budget_enabled():
    return _BUDGET > 0


def plan(T, P, budget=None, layers=None, qkv=None):
    """The cheapest recompute plan for a micro-batch of T k-tokens and P k-patches within budget GiB (defaults: the
    switches' budget, the model's layer counts, OV2_OPT_QKV_RECOMPUTE)."""
    m = MEM_MODEL
    budget = _BUDGET if budget is None else budget
    layers = _LAYERS if layers is None else layers
    qkv = _QKV if qkv is None else qkv
    a0, aT, aP = m['base']
    est = a0 + aT * T + aP * P
    if qkv:                              # always on when set (an elementwise kernel per layer)
        est -= m['qkv_lm'] * T
    p = {'act_lm': False, 'act_vision': False, 'fc1_vision': False, 'fc1_lm': 0, 'full_vision': 0, 'full_lm': 0,
         'est_gib': est}
    steps = [('act_lm', m['act_lm'] * T), ('act_vision', m['act_vision'] * P), ('fc1_vision', m['fc1_vision'] * P)]
    for name, save in steps:
        if est <= budget:
            break
        if save > 0:
            p[name] = True
            est -= save
    for name, per, n in (('fc1_lm', m['fc1_lm_layer'] * T, layers['lm']),
                         ('full_vision', m['full_vision_layer'] * P, layers['vision']),
                         ('full_lm', m['full_lm_layer'] * T, layers['lm'])):
        if est <= budget or per <= 0:
            continue
        k = min(n, int(-(-(est - budget) // per)))
        p[name] = k
        est -= k * per
    p['est_gib'] = round(est, 3)
    return p


def _apply(p):
    s = _STATE
    for cfg in s['cfg_lm']:
        if p['full_lm']:
            cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = 'full', 'block', p['full_lm']
        else:
            cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = s['lm_default']
    for cfg in s['cfg_vision']:
        if p['full_vision']:
            cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = 'full', 'block', p['full_vision']
        else:
            cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = s['vision_default']
    for idx, mlp in s['mlp_vision']:
        mlp._ov2_fc1_recompute = bool(p['fc1_vision']) and idx >= p['full_vision']
        mlp._ov2_act_recompute = bool(p['act_vision']) and idx >= p['full_vision']
    n = len(s['mlp_lm'])
    for idx, mlp in s['mlp_lm']:
        mlp._ov2_fc1_recompute = idx >= max(p['full_lm'], n - p['fc1_lm'])
        mlp._ov2_act_recompute = bool(p['act_lm']) and idx >= p['full_lm']


def install_planner(model):
    """Collect the sub-models' configs and MLPs and plan each training micro-batch in a forward pre-hook."""
    import re
    from megatron.core.transformer.mlp import MLP
    _patch_mlp()
    cfg_lm, cfg_vision, mlp_lm, mlp_vision = {}, {}, [], []
    for name, mod in model.named_modules():
        mm = re.match(r'(vision_model|language_model)\.', name)
        cfg = getattr(mod, 'config', None)
        if mm and cfg is not None and hasattr(cfg, 'recompute_granularity'):
            (cfg_vision if mm.group(1) == 'vision_model' else cfg_lm)[id(cfg)] = cfg
        if isinstance(mod, MLP):
            ml = re.search(r'(vision_model|language_model)\.decoder\.layers\.(\d+)\.mlp$', name)
            if ml:
                (mlp_vision if ml.group(1) == 'vision_model' else mlp_lm).append((int(ml.group(2)), mod))
    first = lambda d: next(iter(d.values()))
    _STATE.update(cfg_lm=list(cfg_lm.values()), cfg_vision=list(cfg_vision.values()), mlp_lm=mlp_lm,
                  mlp_vision=mlp_vision,
                  lm_default=(first(cfg_lm).recompute_granularity, first(cfg_lm).recompute_method,
                              first(cfg_lm).recompute_num_layers),
                  vision_default=(first(cfg_vision).recompute_granularity, first(cfg_vision).recompute_method,
                                  first(cfg_vision).recompute_num_layers),
                  log=os.environ.get('OV2_OPT_MEM_PLAN_LOG', '0') == '1', counts={})

    def pre(mod, args, kwargs):
        if not (mod.training and torch.is_grad_enabled()):
            return
        names = ['images', 'image_grid_thw', 'input_ids']
        d = dict(zip(names, args)); d.update(kwargs)
        ids, img, vid = d.get('input_ids'), d.get('images'), kwargs.get('pixel_values_videos')
        T = (int(ids.numel()) if torch.is_tensor(ids) else 0) / 1000.0
        P = ((int(img.shape[0]) if torch.is_tensor(img) else 0) + (int(vid.shape[0]) if torch.is_tensor(vid) else 0)) / 1000.0
        p = plan(T, P)
        _apply(p)
        if _STATE['log']:
            print(f'[ov2-mem plan] T={T:.3f}k P={P:.3f}k -> {p}', flush=True)
    model.register_forward_pre_hook(pre, with_kwargs=True)
    print(f'[ov2-mem] memory planner: budget {_BUDGET} GiB per micro-batch; {len(cfg_lm)} LM / {len(cfg_vision)} vision '
          f'configs, {len(mlp_lm)} LM / {len(mlp_vision)} vision MLPs', flush=True)
    return model
