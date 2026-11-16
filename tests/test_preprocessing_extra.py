import numpy as np
import pytest

from app.services import preprocessing as pre


class FakeCascade:
    def __init__(self, faces):
        self.faces = faces

    def detectMultiScale(self, *args, **kwargs):  # noqa: N802 - OpenCV's name
        return np.array(self.faces, dtype=int).reshape(-1, 4)


@pytest.fixture
def frame():
    return np.random.default_rng(1).integers(0, 255, (100, 120, 3), dtype=np.uint8)


def test_crop_uses_largest_face_with_margin(frame, monkeypatch):
    monkeypatch.setattr(pre, "_get_face_cascade", lambda: FakeCascade([[5, 5, 10, 10], [20, 20, 40, 40]]))
    crop = pre.crop_face(frame, margin=0.25)
    assert crop.shape[:2] == (60, 60)  # 40px face + 10px margin on each side
    assert np.array_equal(crop, frame[10:70, 10:70])


def test_crop_is_clamped_to_the_frame(frame, monkeypatch):
    monkeypatch.setattr(pre, "_get_face_cascade", lambda: FakeCascade([[0, 0, 40, 40]]))
    assert pre.crop_face(frame, margin=0.25).shape[:2] == (50, 50)  # no negative start, margin only on far side


def test_no_face_returns_the_frame_unchanged(frame, monkeypatch):
    monkeypatch.setattr(pre, "_get_face_cascade", lambda: FakeCascade([]))
    assert pre.crop_face(frame) is frame


def test_preprocess_with_face_detection_resizes_the_crop(frame, monkeypatch):
    monkeypatch.setattr(pre, "_get_face_cascade", lambda: FakeCascade([[20, 20, 40, 40]]))
    batch = pre.preprocess_frames([frame, frame], size=32, mode="unit", detect_faces=True)
    assert batch.shape == (2, 32, 32, 3) and batch.dtype == np.float32


def test_normalization_ranges():
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    black = np.zeros((4, 4, 3), dtype=np.uint8)
    assert pre.normalize_frame(white, "unit").max() == pytest.approx(1.0)
    assert pre.normalize_frame(white, "tanh").max() == pytest.approx(1.0)
    assert pre.normalize_frame(black, "tanh").min() == pytest.approx(-1.0)


def test_imagenet_normalization_centers_on_the_mean():
    mean_pixel = np.full((4, 4, 3), 0, dtype=np.uint8)
    mean_pixel[..., 0], mean_pixel[..., 1], mean_pixel[..., 2] = 124, 116, 104  # ~ (0.485, 0.456, 0.406) * 255
    assert np.abs(pre.normalize_frame(mean_pixel, "imagenet")).max() < 0.05


def test_unknown_normalize_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown normalize mode"):
        pre.normalize_frame(np.zeros((2, 2, 3), dtype=np.uint8), "zscore")


def test_resize_output_shape(frame):
    assert pre.resize_frame(frame, 64).shape == (64, 64, 3)
