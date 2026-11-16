from pathlib import Path

from app.core.config import settings
from app.models.heuristic import HeuristicPredictor
from app.services.pipeline import run_pipeline


def test_pipeline_result_shape(sample_video):
    res = run_pipeline(sample_video, HeuristicPredictor(), num_frames=5)
    assert res.num_frames == 5
    assert len(res.prediction.frame_scores) == 5
    assert res.video.frame_count > 0
    assert res.saved_frames == []


def test_pipeline_uses_settings_when_args_are_omitted(sample_video, monkeypatch):
    monkeypatch.setattr(settings, "num_frames", 3)
    assert run_pipeline(sample_video, HeuristicPredictor()).num_frames == 3


def test_pipeline_can_save_frames(sample_video, tmp_path):
    out = tmp_path / "saved"
    res = run_pipeline(sample_video, HeuristicPredictor(), num_frames=3, save=True, frames_out=out)
    assert len(res.saved_frames) == 3
    assert all(Path(p).is_file() and Path(p).parent == out for p in res.saved_frames)


def test_pipeline_saves_under_frames_dir_when_enabled_in_settings(sample_video, monkeypatch):
    monkeypatch.setattr(settings, "save_frames", True)
    res = run_pipeline(sample_video, HeuristicPredictor(), num_frames=2)
    assert len(res.saved_frames) == 2
    assert Path(res.saved_frames[0]).parent == settings.frames_dir / Path(sample_video).stem
