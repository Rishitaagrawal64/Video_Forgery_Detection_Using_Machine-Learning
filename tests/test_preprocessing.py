import numpy as np
import pytest

from app.services import frame_extractor as fx
from app.services.preprocessing import normalize_frame, preprocess_frames


def test_batch_shape_and_range(sample_video):
    frames = fx.extract_uniform(sample_video, 6)
    batch = preprocess_frames(frames, size=64)
    assert batch.shape == (6, 64, 64, 3)
    assert batch.dtype == np.float32
    assert 0.0 <= batch.min() and batch.max() <= 1.0


def test_normalize_modes():
    f = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert normalize_frame(f, "unit").max() == pytest.approx(1.0)
    assert normalize_frame(f, "tanh").max() == pytest.approx(1.0)
    assert normalize_frame(np.zeros((4, 4, 3), np.uint8), "tanh").min() == pytest.approx(-1.0)
    assert normalize_frame(f, "imagenet").shape == (4, 4, 3)
    with pytest.raises(ValueError):
        normalize_frame(f, "bogus")


def test_face_detection_falls_back_when_no_face(sample_video):
    frames = fx.extract_uniform(sample_video, 2)
    batch = preprocess_frames(frames, size=32, detect_faces=True)
    assert batch.shape == (2, 32, 32, 3)


def test_empty_frames_rejected():
    with pytest.raises(ValueError):
        preprocess_frames([])
