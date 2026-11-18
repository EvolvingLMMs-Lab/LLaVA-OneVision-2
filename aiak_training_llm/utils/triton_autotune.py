"""Cap Triton autotuning to a single config (fla GDN / l2norm kernels).

Every new packed sequence shape triggers a fresh Triton autotune for the fla kernels.
Autotuning compiles and benchmarks every candidate config (up to ~25 for
``l2norm_bwd_kernel``). When a new shape first shows up mid-run, the GPU is already
nearly full from the backward pass, so loading those candidate binaries fails with
``Triton Error [CUDA]: out of memory`` and kills the run (seen in s3 at iteration 207,
a 25-config sweep of ``l2norm_bwd_kernel``). With packed variable-length data a new
shape can appear at any step, so this is not only a warm-up cost.

The fix is to hook the base ``triton.runtime.autotuner.Autotuner.run`` and reduce the
candidate list to one config before tuning. fla's own autotuner subclass calls
``super().run()``, so this single hook covers ``triton.autotune`` and fla's cached
autotuner. It also works regardless of import order, because it patches the class
rather than kernel instances. The config kept is the first with ``num_warps`` in
4, 8, 2, 16, 1, 32 (in that preference order), else the middle one.

Enabled by default in the training entry point; set ``TRITON_AUTOTUNE_CAP=0`` to
restore full autotuning.
"""

import os
import sys

_TARGET_NUM_WARPS = (4, 8, 2, 16, 1, 32)


def _pick_one(configs):
    configs = list(configs)
    if len(configs) <= 1:
        return configs
    for target in _TARGET_NUM_WARPS:
        for c in configs:
            if getattr(c, "num_warps", 4) == target:
                return [c]
    return [configs[len(configs) // 2]]


def apply_triton_autotune_cap() -> bool:
    """Patch Autotuner.run once per process. Returns True if the cap is active."""
    if os.environ.get("TRITON_AUTOTUNE_CAP", "1") == "0":
        return False
    try:
        from triton.runtime.autotuner import Autotuner
    except ImportError:
        return False
    # Same function name as the legacy sitecustomize shim, so the two never stack.
    if getattr(Autotuner.run, "__name__", "") == "_capped_run":
        return True

    orig_run = Autotuner.run

    def _capped_run(self, *args, **kwargs):
        configs = getattr(self, "configs", None)
        if configs is not None and len(configs) > 1:
            self.configs = _pick_one(configs)
        return orig_run(self, *args, **kwargs)

    Autotuner.run = _capped_run
    sys.stderr.write("[triton_autotune] Autotuner.run capped to a single config "
                     "(TRITON_AUTOTUNE_CAP=0 to disable)\n")
    return True
