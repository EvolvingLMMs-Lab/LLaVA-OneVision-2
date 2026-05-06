"""Extract video frames from JSONL samples and backfill frame metadata.

Input:
    JSONL file where each line is a sample containing optional `images_source`
    (list of video paths).

Output:
    JSONL file with `images`, `patch_positions`, and `fps` backfilled,
    plus extracted frame images, `patch_positions.npy`, and `meta.json` saved
    under the output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Any

import numpy as np


# Use software decode and tolerate broken packets so long-running jobs can keep
# processing when encountering partially corrupted H264/AV1 videos.
# os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "hwaccel;none|fflags;+discardcorrupt")
# Reduce FFmpeg decoder log noise (warnings stay hidden, errors still visible).
# os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")


try:
    import cv2
except ImportError as exc:  # pragma: no cover - 运行时提示
    raise SystemExit("缺少依赖 cv2，请先安装 opencv-python。") from exc


def format_fps(fps: float, decimals: int | None = None) -> int | float:
    if decimals is None:
        return int(round(float(fps)))
    return round(float(fps), decimals)


# 3. **抽帧策略**: 参考切图主程序 `run_cut_frames.py` 的默认策略：
#   - 视频时长 < 10 秒：抽 8 帧
#   - 视频时长 < 30 秒：抽 16 帧
#   - 其它更长视频：最多抽 max_frames（默认 32）


@dataclass
class VideoExtractionResult:
    """Container for extracted frames and metadata of a single video.

    Attributes:
        images: Saved frame image paths.
        patch_positions_path: Path to `patch_positions.npy` for this video.
        fps: Frames per second for the source video.
        frame_indices: Selected frame indices in the source video.
    """

    images: list[str]
    patch_positions_path: str
    fps: int | float
    frame_indices: list[int]


def write_video_metadata(
    meta_path: str,
    images: list[str],
    patch_positions_path: str,
    fps: int | float,
    frame_indices: list[int],
) -> None:
    """Persist per-video extraction metadata to JSON.

    Args:
        meta_path: Path to the metadata JSON file.
        images: Saved frame image paths.
        patch_positions_path: Path to `patch_positions.npy`.
        fps: Frames per second for the source video.
        frame_indices: Selected frame indices in the source video.

    Returns:
        None.
    """
    payload = {
        "images": images,
        "patch_positions": patch_positions_path,
        "fps": fps,
        "frame_indices": frame_indices,
    }
    try:
        ensure_dir(os.path.dirname(meta_path))
        tmp_path = f"{meta_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as meta_f:
            json.dump(payload, meta_f, ensure_ascii=False)
        os.replace(tmp_path, meta_path)
    except FileNotFoundError:
        logging.warning("元数据写入失败，路径不存在: %s", meta_path)
    except OSError as exc:
        logging.warning("元数据写入失败: %s (%s)", meta_path, exc)


def load_video_metadata(meta_path: str) -> dict[str, Any] | None:
    """Load per-video extraction metadata from JSON.

    Args:
        meta_path: Path to the metadata JSON file.

    Returns:
        Parsed metadata dict, or None if the file does not exist.
    """
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as meta_f:
            return json.load(meta_f)
    except (json.JSONDecodeError, OSError):
        logging.warning("meta.json 损坏，将重新抽帧: %s", meta_path)
        return None


def choose_target_frames(duration_seconds: float, max_frames: int) -> int:
    """Choose target frame count based on video duration in seconds.

    Args:
        duration_seconds: Video duration in seconds.
        max_frames: Maximum number of frames to extract for long videos.

    Returns:
        Target number of frames to extract.
    """
    if duration_seconds < 10:
        return 8
    if duration_seconds < 30:
        return 16
    return max_frames


def select_frame_indices(frame_count: int, target_count: int) -> list[int]:
    """Select evenly spaced frame indices for a target count.

    Args:
        frame_count: Total number of frames in the video.
        target_count: Desired number of frames to sample.

    Returns:
        List of selected frame indices.
    """
    if frame_count <= target_count:
        return list(range(frame_count))
    return np.linspace(0, frame_count - 1, target_count, dtype=int).tolist()


def select_frame_indices_by_fps(
    frame_count: int,
    fps: float,
    sample_fps: float,
    max_frames: int,
) -> list[int]:
    """Select frame indices by target sampling fps, then cap by max_frames.

    The method first creates timestamp-based indices at fixed intervals
    (e.g. 1 FPS means one frame per second), then uniformly downsamples if the
    sampled indices exceed max_frames.
    """
    if frame_count <= 0:
        return []

    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be > 0, got {sample_fps}")

    source_fps = float(fps) if fps and fps > 0 else 30.0
    duration = frame_count / source_fps

    # Use timestamps for stable cadence under non-integer source fps.
    timestamps = np.arange(0.0, duration, 1.0 / sample_fps, dtype=np.float64)
    indices = np.floor(timestamps * source_fps).astype(np.int64)
    indices = np.clip(indices, 0, frame_count - 1)
    indices = np.unique(indices).tolist()

    if not indices:
        indices = [0]

    if len(indices) <= max_frames:
        return indices

    keep_positions = select_frame_indices(len(indices), max_frames)
    return [indices[pos] for pos in keep_positions]


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int | None = 56 * 56,
    max_pixels: int | None = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if max_pixels and h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif min_pixels and h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def build_patch_positions_for_frame(
    frame_index: int,
    height: int,
    width: int,
    patch_size: int,
    min_pixels: int | None,
    max_pixels: int | None,
    factor_multiplier: int = 2,
) -> np.ndarray:
    """Build (t, h, w) patch indices for a single frame.

    Args:
        frame_index: Index of the frame in the source video.
        height: Frame height.
        width: Frame width.
        patch_size: Patch size to align to.
        min_pixels: Optional minimum pixel constraint.
        max_pixels: Optional maximum pixel constraint.

    Returns:
        NumPy array of shape (num_patches, 3) with (t, h, w) indices.
    """
    resized_h, resized_w = smart_resize(
        height, width, factor=patch_size * factor_multiplier, min_pixels=min_pixels, max_pixels=max_pixels
    )
    tokens_h = resized_h // patch_size
    tokens_w = resized_w // patch_size
    frame_tokens = tokens_h * tokens_w

    per = np.arange(frame_tokens, dtype=np.int64)
    h_positions = per // tokens_w
    w_positions = per % tokens_w
    t_positions = np.full((frame_tokens,), frame_index, dtype=np.int64)
    return np.stack([t_positions, h_positions, w_positions], axis=1)


def iter_jsonl(path: str):
    """Yield JSON objects from a JSONL file path.

    Args:
        path: Input JSONL path.

    Yields:
        Parsed JSON objects per line.
    """
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        None.
    """
    os.makedirs(path, exist_ok=True)


def resolve_output_video_dir(video_path: str, output_root: str, strip_prefixes: list[str]) -> str:
    """Resolve output directory for a video path, optionally stripping prefixes.

    Args:
        video_path: Source video path.
        output_root: Root directory for outputs.
        strip_prefixes: Prefixes to strip from the video path.

    Returns:
        Output directory path for the video.
    """
    normalized = os.path.normpath(video_path)
    for prefix in strip_prefixes:
        prefix_norm = os.path.normpath(prefix)
        if normalized.startswith(prefix_norm):
            rel = normalized[len(prefix_norm) :].lstrip(os.sep)
            return os.path.join(output_root, rel)
    return os.path.join(output_root, normalized.lstrip(os.sep))


def normalize_image_placeholders(messages: list[dict], image_count: int) -> None:
    """Normalize media placeholders in messages.

    This function removes stale `<image>` and `<video>` placeholders from all
    messages, then prepends the expected `<image>` block to the first user
    message when `image_count > 0`.

    Args:
        messages: Message list to update in place.
        image_count: Number of image placeholders to inject.

    Returns:
        None.
    """
    if not messages:
        return

    cleaned_messages = []
    for msg in messages:
        content = msg.get("content", "")
        if "<image>" in content or "<video>" in content:
            content = content.replace("<image>", "")
            content = content.replace("<video>", "")
            content = "\n".join([line for line in content.splitlines() if line.strip()])
        cleaned_messages.append({**msg, "content": content})

    if image_count <= 0:
        messages[:] = cleaned_messages
        return

    placeholder_block = "\n".join(["<image>"] * image_count)
    for idx, msg in enumerate(cleaned_messages):
        if msg.get("role") == "user":
            text = msg.get("content", "").strip()
            msg["content"] = f"{placeholder_block}\n{text}" if text else placeholder_block
            cleaned_messages[idx] = msg
            break

    messages[:] = cleaned_messages


def _extract_frames_via_ffmpeg(
    video_path: str,
    video_dir: str,
    selected_indices: list[int],
    fps: float,
    image_ext: str,
    patch_size: int,
    min_pixels: int | None,
    max_pixels: int | None,
    meta_path: str,
    fps_decimals: int | None = None,
    factor_multiplier: int = 2,
) -> tuple[list[str], list[int], list[np.ndarray]]:
    """Extract specific frames from a video using ffmpeg (fallback for AV1/unsupported codecs).

    Uses a temporary directory to dump all needed frames via ffmpeg's select filter,
    then processes them with cv2 for resizing and patch position building.
    """
    images: list[str] = []
    frame_indices: list[int] = []
    patch_positions_list: list[np.ndarray] = []

    if not selected_indices:
        return images, frame_indices, patch_positions_list

    # Build ffmpeg select expression: select='eq(n\,10)+eq(n\,50)+...'
    select_expr = "+".join(f"eq(n\\,{idx})" for idx in selected_indices)
    with tempfile.TemporaryDirectory(dir="/ov2/tmp_ffmpeg_frames") as tmpdir:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-vf",
            f"select='{select_expr}'",
            "-vsync",
            "0",
            os.path.join(tmpdir, "frame_%05d.png"),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logging.warning("ffmpeg fallback 失败: %s  err=%s", video_path, exc)
            return images, frame_indices, patch_positions_list

        # ffmpeg outputs frames numbered 00001, 00002, ... in order of selected_indices
        tmp_files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".png"))
        for tmp_name, frame_idx in zip(tmp_files, selected_indices):
            tmp_path = os.path.join(tmpdir, tmp_name)
            frame = cv2.imread(tmp_path)
            if frame is None:
                continue

            resized_h, resized_w = smart_resize(
                frame.shape[0],
                frame.shape[1],
                factor=patch_size * factor_multiplier,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if (resized_h, resized_w) != (frame.shape[0], frame.shape[1]):
                interpolation = (
                    cv2.INTER_AREA if resized_h < frame.shape[0] or resized_w < frame.shape[1] else cv2.INTER_LINEAR
                )
                frame = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

            frame_name = f"frame_{frame_idx:05d}{image_ext}"
            frame_path = os.path.join(video_dir, frame_name)
            if not cv2.imwrite(frame_path, frame):
                raise RuntimeError(f"写入图片失败: {frame_path}")

            images.append(frame_path)
            frame_indices.append(frame_idx)
            patch_positions_list.append(
                build_patch_positions_for_frame(frame_idx, resized_h, resized_w, patch_size, min_pixels, max_pixels, factor_multiplier)
            )
            write_video_metadata(
                meta_path,
                images=images,
                patch_positions_path=os.path.join(video_dir, "patch_positions.npy"),
                fps=format_fps(fps, fps_decimals),
                frame_indices=frame_indices,
            )

    return images, frame_indices, patch_positions_list


def extract_video_frames(
    video_path: str,
    output_root: str,
    image_ext: str,
    patch_size: int,
    min_pixels: int | None,
    max_pixels: int | None,
    max_frames: int,
    sample_fps: float | None,
    strip_prefixes: list[str],
    timeout_seconds: int,
    fps_decimals: int | None = None,
    factor_multiplier: int = 2,
) -> VideoExtractionResult | None:
    """Extract frames from a video and build patch positions.

    Args:
        video_path: Path to the input video.
        output_root: Root directory for extracted frames and metadata.
        image_ext: Output image extension.
        patch_size: Patch size used for position encoding.
        min_pixels: Minimum pixel constraint for resizing.
        max_pixels: Maximum pixel constraint for resizing.
        max_frames: Maximum number of frames to extract for long videos.
        sample_fps: Optional target sampling FPS. If set, frames are sampled at
            this cadence before applying max_frames cap.
        strip_prefixes: Prefixes to strip when constructing output directories.
        timeout_seconds: Max allowed extraction time per video.

    Returns:
        VideoExtractionResult containing frame image paths, patch position file,
        fps, and selected frame indices. Returns None on timeout or if no frames
        are extracted.
    """
    images: list[str] = []
    frame_indices: list[int] = []
    patch_positions_list: list[np.ndarray] = []
    start_time = time.monotonic()

    video_dir = resolve_output_video_dir(video_path, output_root, strip_prefixes)
    ensure_dir(video_dir)
    meta_path = os.path.join(video_dir, "meta.json")

    # Check if extraction already exists
    existing_meta = load_video_metadata(meta_path)
    if existing_meta:
        # Verify all referenced files exist
        existing_images = existing_meta.get("images", [])
        patch_path = existing_meta.get("patch_positions")
        all_files_exist = (
            isinstance(existing_images, list)
            and existing_images
            and isinstance(patch_path, str)
            and os.path.exists(patch_path)
            and all(os.path.exists(img) for img in existing_images)
        )
        if all_files_exist:
            existing_fps = existing_meta.get("fps")
            if not isinstance(existing_fps, (int, float)) or existing_fps <= 0:
                existing_fps = 30
            # Frames already extracted, skip re-extraction
            return VideoExtractionResult(
                images=existing_images,
                patch_positions_path=patch_path,
                fps=format_fps(existing_fps, fps_decimals),
                frame_indices=existing_meta.get("frame_indices", []),
            )

    # logging.info("开始抽帧: %s", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.warning("无法打开视频，已跳过: %s", video_path)
        shutil.rmtree(video_dir, ignore_errors=True)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def timed_out() -> bool:
        return (time.monotonic() - start_time) > timeout_seconds

    if frame_count > 0:
        if sample_fps is not None:
            selected_indices = select_frame_indices_by_fps(frame_count, fps, sample_fps, max_frames)
        else:
            duration = frame_count / fps
            target_count = choose_target_frames(duration, max_frames)
            selected_indices = select_frame_indices(frame_count, target_count)

        for frame_idx in selected_indices:
            if timed_out():
                cap.release()
                shutil.rmtree(video_dir, ignore_errors=True)
                logging.warning("抽帧超时，已放弃: %s", video_path)
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            resized_h, resized_w = smart_resize(
                frame.shape[0],
                frame.shape[1],
                factor=patch_size * factor_multiplier,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if (resized_h, resized_w) != (frame.shape[0], frame.shape[1]):
                interpolation = (
                    cv2.INTER_AREA if resized_h < frame.shape[0] or resized_w < frame.shape[1] else cv2.INTER_LINEAR
                )
                frame = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

            frame_name = f"frame_{frame_idx:05d}{image_ext}"
            frame_path = os.path.join(video_dir, frame_name)
            if not cv2.imwrite(frame_path, frame):
                raise RuntimeError(f"写入图片失败: {frame_path}")

            images.append(frame_path)
            frame_indices.append(frame_idx)
            patch_positions_list.append(
                build_patch_positions_for_frame(
                    frame_idx,
                    resized_h,
                    resized_w,
                    patch_size,
                    min_pixels,
                    max_pixels,
                    factor_multiplier,
                )
            )
            write_video_metadata(
                meta_path,
                images=images,
                patch_positions_path=os.path.join(video_dir, "patch_positions.npy"),
                fps=format_fps(fps, fps_decimals),
                frame_indices=frame_indices,
            )
    else:
        temp_frames: list[tuple[int, str, int, int]] = []
        frame_idx = 0
        while True:
            if timed_out():
                cap.release()
                shutil.rmtree(video_dir, ignore_errors=True)
                logging.warning("抽帧超时，已放弃: %s", video_path)
                return None
            ret, frame = cap.read()
            if not ret:
                break

            resized_h, resized_w = smart_resize(
                frame.shape[0],
                frame.shape[1],
                factor=patch_size * factor_multiplier,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            if (resized_h, resized_w) != (frame.shape[0], frame.shape[1]):
                interpolation = (
                    cv2.INTER_AREA if resized_h < frame.shape[0] or resized_w < frame.shape[1] else cv2.INTER_LINEAR
                )
                frame = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)

            frame_name = f"frame_tmp_{frame_idx:05d}{image_ext}"
            frame_path = os.path.join(video_dir, frame_name)
            if not cv2.imwrite(frame_path, frame):
                raise RuntimeError(f"写入图片失败: {frame_path}")

            temp_frames.append((frame_idx, frame_path, resized_h, resized_w))
            frame_idx += 1

        if sample_fps is not None:
            selected = set(select_frame_indices_by_fps(frame_idx, fps, sample_fps, max_frames))
        else:
            duration = frame_idx / fps
            target_count = choose_target_frames(duration, max_frames)
            selected = set(select_frame_indices(frame_idx, target_count))

        for src_idx, src_path, src_h, src_w in temp_frames:
            if timed_out():
                cap.release()
                shutil.rmtree(video_dir, ignore_errors=True)
                logging.warning("抽帧超时，已放弃: %s", video_path)
                return None
            if src_idx not in selected:
                os.remove(src_path)
                continue

            frame_name = f"frame_{src_idx:05d}{image_ext}"
            frame_path = os.path.join(video_dir, frame_name)
            os.replace(src_path, frame_path)

            images.append(frame_path)
            frame_indices.append(src_idx)
            patch_positions_list.append(
                build_patch_positions_for_frame(
                    src_idx,
                    src_h,
                    src_w,
                    patch_size,
                    min_pixels,
                    max_pixels,
                    factor_multiplier,
                )
            )
            write_video_metadata(
                meta_path,
                images=images,
                patch_positions_path=os.path.join(video_dir, "patch_positions.npy"),
                fps=format_fps(fps, fps_decimals),
                frame_indices=frame_indices,
            )

    cap.release()

    patch_path = os.path.join(video_dir, "patch_positions.npy")
    if patch_positions_list:
        patch_positions_array = np.concatenate(patch_positions_list, axis=0)
    else:
        # cv2 failed to decode any frame — fallback to ffmpeg (e.g. AV1 codec)
        logging.warning("cv2 未能解码，尝试 ffmpeg fallback: %s", video_path)
        if frame_count > 0:
            if sample_fps is not None:
                fb_indices = select_frame_indices_by_fps(frame_count, fps, sample_fps, max_frames)
            else:
                duration = frame_count / fps
                target_count = choose_target_frames(duration, max_frames)
                fb_indices = select_frame_indices(frame_count, target_count)
        else:
            fb_indices = select_frame_indices(100, choose_target_frames(10.0, max_frames))
        images, frame_indices, patch_positions_list = _extract_frames_via_ffmpeg(
            video_path,
            video_dir,
            fb_indices,
            fps,
            image_ext,
            patch_size,
            min_pixels,
            max_pixels,
            meta_path,
            fps_decimals,
            factor_multiplier,
        )
        if not patch_positions_list:
            shutil.rmtree(video_dir, ignore_errors=True)
            logging.warning("未抽取到有效帧，已放弃: %s", video_path)
            return None
        patch_positions_array = np.concatenate(patch_positions_list, axis=0)
    np.save(patch_path, patch_positions_array)

    write_video_metadata(
        meta_path,
        images=images,
        patch_positions_path=patch_path,
        fps=format_fps(fps, fps_decimals),
        frame_indices=frame_indices,
    )
    # logging.info("抽帧完成: %s (frames=%d)", video_path, len(images))

    return VideoExtractionResult(
        images=images,
        patch_positions_path=patch_path,
        fps=format_fps(fps, fps_decimals),
        frame_indices=frame_indices,
    )


def extract_video_worker(
    video_path: str,
    output_dir: str,
    image_ext: str,
    patch_size: int,
    min_pixels: int | None,
    max_pixels: int | None,
    max_frames: int,
    sample_fps: float | None,
    strip_prefixes: list[str],
    fps_decimals: int | None = None,
    factor_multiplier: int = 2,
) -> tuple[str, bool]:
    """Worker wrapper to extract a single video and return keyed result.

    Args:
        video_path: Path to the input video.
        output_dir: Root directory for extracted frames and metadata.
        image_ext: Output image extension.
        patch_size: Patch size used for position encoding.
        min_pixels: Minimum pixel constraint for resizing.
        max_pixels: Maximum pixel constraint for resizing.
        max_frames: Maximum number of frames to extract for long videos.
        sample_fps: Optional target sampling FPS.
        strip_prefixes: Prefixes to strip when constructing output directories.
        fps_decimals: Number of decimal places for fps (None = integer).

    Returns:
        A tuple of the original video path and success flag.
    """
    result = extract_video_frames(
        video_path=video_path,
        output_root=output_dir,
        image_ext=image_ext,
        patch_size=patch_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_frames=max_frames,
        sample_fps=sample_fps,
        strip_prefixes=strip_prefixes,
        timeout_seconds=480,
        fps_decimals=fps_decimals,
        factor_multiplier=factor_multiplier,
    )
    return video_path, result is not None


def _process_output_item(
    item: dict,
    output_dir: str,
    strip_prefix: list[str],
    fps_decimals: int | None = None,
) -> str | None:
    """Load meta, compute fps, and serialize one JSONL item.

    Args:
        item: Parsed JSONL sample dict.
        output_dir: Root directory that contains extracted frame data.
        strip_prefix: Prefixes to strip when resolving video output dirs.
        fps_decimals: Number of decimal places for fps (None = integer).

    Returns:
        JSON-serialized line string ready to write, or None if the item
        has no valid extracted frames and should be skipped.
    """
    video_paths = item.get("images_source")
    if not video_paths:
        return json.dumps(item, ensure_ascii=False)

    images: list[str] = []
    patch_positions: list[str] = []
    timestamps: dict[str, str] = {}

    for video_path in video_paths:
        video_dir = resolve_output_video_dir(video_path, output_dir, strip_prefix)
        meta_path = os.path.join(video_dir, "meta.json")
        meta = load_video_metadata(meta_path)
        if not meta:
            return None

        meta_images = [img for img in meta.get("images", []) if os.path.exists(img)]
        if not meta_images:
            return None

        offset = len(images)
        images.extend(meta_images)
        patch_path = meta.get("patch_positions")
        if isinstance(patch_path, str) and os.path.exists(patch_path):
            patch_positions.append(patch_path)

        meta_timestamps = meta.get("timestamps", {})
        meta_indices = meta.get("frame_indices", [])
        if isinstance(meta_timestamps, dict) and meta_indices:
            for meta_idx, frame_idx in enumerate(meta_indices):
                timestamp_key = str(frame_idx)
                if timestamp_key in meta_timestamps:
                    timestamps[str(offset + meta_idx)] = meta_timestamps[timestamp_key]

    item["images"] = images
    item["patch_positions"] = patch_positions
    item["timestamp"] = timestamps

    fps = item.get("fps")
    if fps is None and video_paths:
        try:
            cap = cv2.VideoCapture(video_paths[0])
            if not cap.isOpened():
                raise ValueError(f"Failed to open video for fps extraction: {video_paths[0]}")
            _fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if _fps is None or _fps <= 0:
                raise ValueError(f"Failed to read valid fps from video: {video_paths[0]}")
            fps = _fps
        except Exception as exc:
            logging.warning(
                "Failed to infer fps from %s (%s): %s",
                video_paths[0],
                type(exc).__name__,
                exc,
            )
    if fps is not None and fps > 0:
        item["fps"] = format_fps(fps, fps_decimals)

    normalize_image_placeholders(item.get("messages", []), len(images))

    return json.dumps(item, ensure_ascii=False)


def main() -> None:
    """CLI entry: extract frames for all videos and write updated JSONL.

    Input:
        --input-jsonl: JSONL file with `images_source` entries.

    Output:
        --output-jsonl: JSONL file with `images`, `patch_positions`, `fps`.
        --output-dir: Directory containing extracted frame images and
        `patch_positions.npy` and `meta.json` per video.
    """
    parser = argparse.ArgumentParser(description="从 jsonl 中提取视频帧并回填字段")
    parser.add_argument("--input-jsonl", required=True, help="输入 jsonl 路径")
    parser.add_argument("--output-jsonl", required=True, help="输出 jsonl 路径")
    parser.add_argument("--output-dir", required=True, help="帧图片输出根目录")
    parser.add_argument("--image-ext", default=".jpg", help="输出图片扩展名，默认 .jpg")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="并行进程数，默认 CPU 核心数",
    )
    parser.add_argument("--chunksize", type=int, default=32, help="多进程任务分块大小，默认 32")
    parser.add_argument("--patch-size", type=int, default=14, help="patch 大小，默认 14")
    parser.add_argument(
        "--factor-multiplier",
        type=int,
        default=2,
        help="smart_resize 的 factor = patch_size * factor_multiplier，默认 2",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=56 * 56,
        help="smart_resize 的最小像素数，默认 56*56",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=768 * 768,
        help="smart_resize 的最大分辨率（像素数上限），默认 768*768",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="视频最大抽帧数（长视频上限），默认 32",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="按目标 FPS 抽帧（如 1 表示 1FPS），再由 --max-frames 限制上限",
    )
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=[],
        help="输出路径需裁剪的前缀（可多次传入）",
    )
    parser.add_argument(
        "--fps-decimals",
        type=int,
        default=None,
        help="fps 字段保留的小数位数（不传则取整）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    ensure_dir(args.output_dir)

    items = list(iter_jsonl(args.input_jsonl))
    unique_videos: dict[str, None] = {}
    for item in items:
        for video_path in item.get("images_source", []) or []:
            unique_videos[video_path] = None

    total_videos = len(unique_videos)
    logging.info("待处理视频数: %d", total_videos)

    worker_fn = partial(
        extract_video_worker,
        output_dir=args.output_dir,
        image_ext=args.image_ext,
        patch_size=args.patch_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
        sample_fps=args.sample_fps,
        strip_prefixes=args.strip_prefix,
        fps_decimals=args.fps_decimals,
        factor_multiplier=args.factor_multiplier,
    )

    done_count = 0
    ok_count = 0
    with Pool(processes=args.num_workers) as pool:
        for _video_path, _ok in pool.imap_unordered(
            worker_fn,
            unique_videos.keys(),
            chunksize=args.chunksize,
        ):
            done_count += 1
            if _ok:
                ok_count += 1
            if done_count % 100 == 0 or done_count == total_videos:
                logging.info("抽帧进度: %d/%d (成功 %d)", done_count, total_videos, ok_count)

    with open(args.output_jsonl, "w", buffering=1 << 20) as out_f:
        total_items = len(items)
        for idx, item in enumerate(items, start=1):
            video_paths = item.get("images_source")
            if not video_paths:
                normalize_image_placeholders(item.get("messages", []), 0)
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            images: list[str] = []
            patch_positions: list[str] = []
            fps_values: list[int | float] = []
            valid_sample = True

            for video_path in video_paths:
                video_dir = resolve_output_video_dir(video_path, args.output_dir, args.strip_prefix)
                meta_path = os.path.join(video_dir, "meta.json")
                meta = load_video_metadata(meta_path)
                if not meta:
                    valid_sample = False
                    break

                meta_images = [img for img in meta.get("images", []) if os.path.exists(img)]
                if not meta_images:
                    valid_sample = False
                    break

                images.extend(meta_images)
                patch_path = meta.get("patch_positions")
                if isinstance(patch_path, str) and os.path.exists(patch_path):
                    patch_positions.append(patch_path)

                meta_fps = meta.get("fps")
                if not isinstance(meta_fps, (int, float)) or meta_fps <= 0:
                    valid_sample = False
                    break
                fps_values.append(format_fps(meta_fps, args.fps_decimals))

            if not valid_sample:
                continue

            item["images"] = images
            item["patch_positions"] = patch_positions
            item.pop("timestamp", None)
            item["fps"] = fps_values[0] if len(fps_values) == 1 else fps_values
            normalize_image_placeholders(item.get("messages", []), len(images))

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if idx % 100 == 0 or idx == total_items:
                logging.info("写出进度: %d/%d", idx, total_items)


if __name__ == "__main__":
    main()
