"""Optional attention kernels that are not part of the training image (docs/faster_training.md).

- FlashAttention-4 (`flash-attn-4`, CuTe DSL kernels for sm90 / sm100, module `flash_attn.cute`): OV2_OPT_FA4_ATTN.
- FlashAttention-3 (`flash_attn_3`, sm90 only, module `flash_attn_interface`): OV2_OPT_FA3_ATTN (experimental).

Both are imported normally. OV2_EXTRAS_DIR=<dir> additionally makes packages installed with
`pip install --no-deps --target <dir>/fa4 ...` (or `<dir>/fa3`) importable, which leaves the image's own Python
environment untouched. The directory is appended after the environment's packages, so the image's torch and
flash-attn 2 stay the ones in use; FA4 only adds the `flash_attn.cute` subpackage. A switch whose library is missing
fails at model build time with the install hint below.
"""
import os
import site
import sys

FA4_PACKAGES = ('flash-attn-4==4.0.0b36 nvidia-cutlass-dsl==4.8.1 nvidia-cutlass-dsl-libs-base==4.8.1 '
                'nvidia-cutlass-dsl-libs-core==4.8.1 nvidia-cutlass-dsl-libs-cu12==4.8.1 apache-tvm-ffi==0.1.15.post1 '
                'quack-kernels==0.6.5 torch-c-dlpack-ext==0.1.5')


def _extras_subdir(name):
    root = os.environ.get('OV2_EXTRAS_DIR')
    return os.path.join(root, name) if root else None


class _MissingDtype:
    """Stands in for a torch dtype this torch does not have (never equal to a real dtype)."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<torch.{self.name}: not in this torch>'


def _torch_dtype_placeholders():
    # quack-kernels names torch.float4_e2m1fn_x2 / float8_e8m0fnu (torch >= 2.8) in its dtype tables at import time.
    # On torch 2.7 a placeholder that equals no torch dtype is set instead: FP4 / MX paths are unusable, nothing else
    # changes.
    import torch
    for name in ('float4_e2m1fn_x2', 'float8_e8m0fnu'):
        if not hasattr(torch, name):
            setattr(torch, name, _MissingDtype(name))


def _typing_extensions_get_annotations():
    # apache-tvm-ffi imports typing_extensions.get_annotations (typing_extensions >= 4.13). If the already loaded
    # typing_extensions is older, provide it from inspect.get_annotations (default `format` only).
    import inspect
    import typing_extensions as te
    if hasattr(te, 'get_annotations'):
        return

    def get_annotations(obj, *, globals=None, locals=None, eval_str=False, format=None):
        if format not in (None, 1):          # 1 = annotationlib.Format.VALUE, the backport's default
            raise NotImplementedError('get_annotations(format=...) needs typing_extensions >= 4.13')
        return inspect.get_annotations(obj, globals=globals, locals=locals, eval_str=eval_str)
    te.get_annotations = get_annotations


def load_fa4():
    """Import FlashAttention-4 and return the `flash_attn.cute` module."""
    _torch_dtype_placeholders()
    extra = _extras_subdir('fa4')
    if extra and os.path.isdir(extra) and extra not in sys.path:
        site.addsitedir(extra)               # appended; also runs nvidia-cutlass-dsl's .pth (its `cutlass` package)
    _typing_extensions_get_annotations()
    try:
        import flash_attn
        sub = os.path.join(extra, 'flash_attn') if extra else None
        if sub and os.path.isdir(sub) and sub not in list(flash_attn.__path__):
            # flash-attn 2 owns the `flash_attn` package; FA4 installed elsewhere is its `cute` subpackage
            flash_attn.__path__.append(sub)
        import flash_attn.cute
    except ImportError as e:
        raise RuntimeError(
            f'OV2_OPT_FA4_ATTN=1 needs FlashAttention-4 ({e}). Install it next to the image\'s torch and '
            f'flash-attn 2 with `pip install --no-deps --target $OV2_EXTRAS_DIR/fa4 {FA4_PACKAGES}` and set '
            f'OV2_EXTRAS_DIR; see docs/faster_training.md') from e
    return flash_attn.cute


def load_fa3():
    """Import FlashAttention-3 (Hopper) and return the `flash_attn_interface` module."""
    extra = _extras_subdir('fa3')
    if extra and os.path.isdir(extra) and extra not in sys.path:
        sys.path.append(extra)
    try:
        import flash_attn_interface
    except ImportError as e:
        raise RuntimeError(
            f'OV2_OPT_FA3_ATTN needs FlashAttention-3 ({e}): build it with tools/kernels/build_fa3.sh into '
            f'$OV2_EXTRAS_DIR/fa3 and set OV2_EXTRAS_DIR; see docs/faster_training.md') from e
    return flash_attn_interface
