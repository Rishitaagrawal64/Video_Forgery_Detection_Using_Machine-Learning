"""Frame extraction module (Sprint 1).

Two sampling strategies are provided:
  * uniform  - N frames evenly spaced across the whole video (best for classification)
  * by_fps   - one frame every 1/target_fps seconds, capped at max_frames
Frames are returned as RGB uint8 arrays of shape (H, W, 3).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.exceptions import EmptyVideoError, VideoOpenError


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


@contextmanager
def _open(path: str | Path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoOpenError(f"Cannot open video: {path}")
        yield cap
    finally:
        cap.release()


def get_video_info(path: str | Path) -> VideoInfo:
    with _open(path) as cap:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = count / fps if fps > 0 else 0.0
    return VideoInfo(str(path), width, height, fps, count, duration)


def _read_at(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def extract_uniform(path: str | Path, num_frames: int = 16) -> list[np.ndarray]:
    """Return `num_frames` evenly spaced RGB frames (fewer if the video is shorter)."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")

    with _open(path) as cap:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return _sequential_fallback(cap, num_frames)

        n = min(num_frames, total)
        indices = np.linspace(0, total - 1, n).round().astype(int)
        frames = [f for f in (_read_at(cap, int(i)) for i in indices) if f is not None]

    if not frames:
        raise EmptyVideoError(f"No readable frames in: {path}")
    return frames


def _sequential_fallback(cap: cv2.VideoCapture, num_frames: int) -> list[np.ndarray]:
    """For containers that don't report a frame count: read all, then subsample."""
    all_frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not all_frames:
        raise EmptyVideoError("No readable frames in video")
    idx = np.linspace(0, len(all_frames) - 1, min(num_frames, len(all_frames))).round().astype(int)
    return [all_frames[i] for i in idx]


def extract_by_fps(
    path: str | Path, target_fps: float = 1.0, max_frames: int | None = 64
) -> list[np.ndarray]:
    """Sample roughly `target_fps` frames per second of video."""
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")

    with _open(path) as cap:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(int(round(src_fps / target_fps)), 1)

        frames: list[np.ndarray] = []
        idx = 0
        while True:
            if total and idx >= total:
                break
            if max_frames is not None and len(frames) >= max_frames:
                break
            frame = _read_at(cap, idx)
            if frame is None:
                break
            frames.append(frame)
            idx += step

    if not frames:
        raise EmptyVideoError(f"No readable frames in: {path}")
    return frames


def save_frames(frames: list[np.ndarray], out_dir: str | Path, prefix: str = "frame") -> list[Path]:
    """Write RGB frames to disk as JPEGs; returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        p = out / f"{prefix}_{i:04d}.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        paths.append(p)
    return paths
