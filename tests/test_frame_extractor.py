import pytest

from app.core.exceptions import VideoOpenError
from app.services import frame_extractor as fx


def test_video_info(sample_video):
    info = fx.get_video_info(sample_video)
    assert (info.width, info.height) == (160, 120)
    assert info.frame_count == 75
    assert info.fps == pytest.approx(25.0)
    assert info.duration_sec == pytest.approx(3.0)


def test_extract_uniform_count_and_shape(sample_video):
    frames = fx.extract_uniform(sample_video, 8)
    assert len(frames) == 8
    assert frames[0].shape == (120, 160, 3)
    assert frames[0].dtype.name == "uint8"


def test_extract_uniform_more_than_available(sample_video):
    assert len(fx.extract_uniform(sample_video, 500)) == 75


def test_extract_by_fps(sample_video):
    frames = fx.extract_by_fps(sample_video, target_fps=5, max_frames=None)
    assert len(frames) == 15  # 3s * 5fps


def test_extract_by_fps_cap(sample_video):
    assert len(fx.extract_by_fps(sample_video, target_fps=25, max_frames=10)) == 10


def test_missing_file_raises(tmp_path):
    with pytest.raises(VideoOpenError):
        fx.extract_uniform(tmp_path / "nope.mp4")


def test_invalid_args(sample_video):
    with pytest.raises(ValueError):
        fx.extract_uniform(sample_video, 0)
    with pytest.raises(ValueError):
        fx.extract_by_fps(sample_video, 0)


def test_save_frames(sample_video, tmp_path):
    frames = fx.extract_uniform(sample_video, 4)
    paths = fx.save_frames(frames, tmp_path / "out")
    assert len(paths) == 4 and all(p.exists() for p in paths)
