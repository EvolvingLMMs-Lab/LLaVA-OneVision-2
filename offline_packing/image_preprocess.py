#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""image_preprocess.py — shared offline image resize/re-encode policy.

Policy: longest_edge=3072; small images keep their bytes; PNG/WebP are forced to
JPEG; enabled by default; EXIF transpose ON.

Public API:
    policy = ImagePolicy(longest_edge=3072, jpeg_quality=92,
                         exif_transpose=True, force_jpeg=True)
    new_bytes, info = policy.apply_to_bytes(orig_bytes, src_name="foo.jpg")

When the image is already small AND already a JPEG, ``apply_to_bytes`` returns
the original bytes unchanged (info["action"] == "keep").
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_JPEG_EXTS = {".jpg", ".jpeg"}
_RECODE_EXTS = {".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ImagePolicy:
    longest_edge: int = 3072
    max_pixels: Optional[int] = None      # alternative cap; if set, also enforced
    jpeg_quality: int = 92
    exif_transpose: bool = True
    force_jpeg: bool = True               # PNG/WebP/etc always recoded to JPEG
    enabled: bool = True

    def _should_resize(self, w: int, h: int) -> bool:
        if w <= 0 or h <= 0:
            return False
        if self.longest_edge and max(w, h) > self.longest_edge:
            return True
        if self.max_pixels and (w * h) > self.max_pixels:
            return True
        return False

    def _compute_new_size(self, w: int, h: int) -> tuple[int, int]:
        scale = 1.0
        if self.longest_edge and max(w, h) > self.longest_edge:
            scale = min(scale, self.longest_edge / float(max(w, h)))
        if self.max_pixels and (w * h) > self.max_pixels:
            scale = min(scale, (self.max_pixels / float(w * h)) ** 0.5)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return new_w, new_h

    def apply_to_bytes(self, data: bytes, src_name: str = "") -> tuple[bytes, dict]:
        info: dict = {
            "action": "keep",
            "src_bytes": len(data),
            "orig_size": None,
            "new_size": None,
        }
        if not self.enabled:
            return data, info

        ext = os.path.splitext(src_name)[1].lower()
        needs_recode = self.force_jpeg and (ext in _RECODE_EXTS)

        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                if self.exif_transpose:
                    im = ImageOps.exif_transpose(im)
                w, h = im.size
                info["orig_size"] = (w, h)

                resize_needed = self._should_resize(w, h)
                fmt = (im.format or "").upper()
                is_jpeg = fmt == "JPEG" or ext in _JPEG_EXTS
                needs_rgb_convert = im.mode not in ("RGB", "L")

                # Fast path: small, already JPEG, no EXIF rotation applied -> keep bytes.
                # We can't cheaply prove "no EXIF rotation was applied", so we
                # conservatively keep bytes only when no resize is needed AND
                # the image is JPEG AND mode is RGB/L AND no recode needed.
                if (not resize_needed) and is_jpeg and (not needs_recode) and (not needs_rgb_convert):
                    # Check if exif_transpose actually changed size — if so, must re-encode.
                    # If orig W/H equals current W/H AND format is jpeg, keep raw bytes.
                    info["new_size"] = (w, h)
                    return data, info

                # Re-encode path.
                if needs_rgb_convert:
                    im = im.convert("RGB")
                if resize_needed:
                    new_w, new_h = self._compute_new_size(w, h)
                    im = im.resize((new_w, new_h), Image.BICUBIC)
                    info["action"] = "resize"
                else:
                    info["action"] = "recode" if (needs_recode or needs_rgb_convert) else "noop_recode"

                out = io.BytesIO()
                im.save(out, format="JPEG", quality=self.jpeg_quality,
                        optimize=False, progressive=False)
                new_bytes = out.getvalue()
                info["new_size"] = im.size
                info["new_bytes"] = len(new_bytes)
                return new_bytes, info
        except Exception as exc:  # noqa: BLE001
            logger.warning("image_preprocess: decode failed for %s: %s -- keeping original bytes",
                           src_name, exc)
            info["action"] = "error"
            info["error"] = str(exc)
            return data, info

    def apply_to_bytes_exact(self, data: bytes, target_w: int, target_h: int,
                             src_name: str = "") -> tuple[bytes, dict]:
        """Resize to exactly (target_w, target_h) and JPEG-encode.

        Used by s4 for master-frame samples: s0 already chose the per-video target
        size (smart_resize, patch*merge aligned) and built patch_positions for it,
        so the packed image must have exactly that size. Unlike apply_to_bytes this
        ignores ``enabled``/longest_edge and raises instead of keeping the original
        bytes on failure, since a wrong-sized frame would not match its
        patch_positions.
        """
        info = {"action": "resize_exact", "src_bytes": len(data), "orig_size": None,
                "new_size": (target_w, target_h)}
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if self.exif_transpose:
                im = ImageOps.exif_transpose(im)
            info["orig_size"] = im.size
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            if im.size != (target_w, target_h):
                im = im.resize((target_w, target_h), Image.BICUBIC)
            out = io.BytesIO()
            im.save(out, format="JPEG", quality=self.jpeg_quality, optimize=False, progressive=False)
            return out.getvalue(), info


def policy_from_args(args, default_enabled: bool = True) -> ImagePolicy:
    """Build a policy from argparse Namespace with attributes:
        longest_edge, jpeg_quality, max_pixels (optional), no_preprocess (optional),
        no_exif_transpose (optional), no_force_jpeg (optional).
    """
    enabled = default_enabled and not getattr(args, "no_preprocess", False)
    return ImagePolicy(
        longest_edge=int(getattr(args, "longest_edge", 3072)),
        jpeg_quality=int(getattr(args, "jpeg_quality", 92)),
        max_pixels=getattr(args, "max_pixels_offline", None),
        exif_transpose=not getattr(args, "no_exif_transpose", False),
        force_jpeg=not getattr(args, "no_force_jpeg", False),
        enabled=enabled,
    )
