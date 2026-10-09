"""End-to-end pipeline: video -> frames -> preprocessed batch -> prediction."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.models.base import BasePredictor, Prediction
from app.services import frame_extractor as fx
from app.services.preprocessing import preprocess_frames


@dataclass
class PipelineResult:
    video: fx.VideoInfo
    num_frames: int
    prediction: Prediction
    saved_frames: list[str] = field(default_factory=list)


def run_pipeline(
    video_path: str | Path,
    predictor: BasePredictor,
    *,
    num_frames: int | None = None,
    size: int | None = None,
    mode: str | None = None,
    detect_faces: bool | None = None,
    save: bool | None = None,
    frames_out: str | Path | None = None,
) -> PipelineResult:
    info = fx.get_video_info(video_path)
    frames = fx.extract_uniform(video_path, num_frames or settings.num_frames)

    saved: list[str] = []
    if settings.save_frames if save is None else save:
        out = Path(frames_out) if frames_out else settings.frames_dir / Path(video_path).stem
        saved = [str(p) for p in fx.save_frames(frames, out)]

    batch = preprocess_frames(
        frames,
        size=size or settings.frame_size,
        mode=mode or settings.normalize_mode,
        detect_faces=settings.detect_faces if detect_faces is None else detect_faces,
    )
    return PipelineResult(info, len(frames), predictor.predict(batch), saved)
