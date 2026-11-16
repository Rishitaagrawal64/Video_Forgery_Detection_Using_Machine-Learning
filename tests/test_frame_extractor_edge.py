"""Edge cases of frame extraction that real files rarely trigger: containers that do not report a frame
count, frames that cannot be decoded, and empty streams. A fake capture object stands in for OpenCV."""
from contextlib import contextmanager

import cv2
import numpy as np
import pytest

from app.core.exceptions import EmptyVideoError
from app.services import frame_extractor as fx


class FakeCap:
    """Minimal cv2.VideoCapture stand-in. Frame i is a uniform image with value i (BGR == RGB)."""

    def __init__(self, n_frames: int, reported_count: int = 0, fps: float = 25.0):
        self.frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n_frames)]
        self.reported_count, self.fps, self.pos = reported_count, fps, 0

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_COUNT: self.reported_count, cv2.CAP_PROP_FPS: self.fps}.get(prop, 0)

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.pos = int(value)

    def read(self):
        if 0 <= self.pos < len(self.frames):
            frame = self.frames[self.pos]
            self.pos += 1
            return True, frame
        return False, None


@pytest.fixture
def use_cap(monkeypatch):
    def install(cap: FakeCap) -> None:
        @contextmanager
        def fake_open(_path):
            yield cap

        monkeypatch.setattr(fx, "_open", fake_open)

    return install


def values(frames) -> list[int]:
    return [int(f[0, 0, 0]) for f in frames]


def test_unknown_frame_count_falls_back_to_sequential_read(use_cap):
    use_cap(FakeCap(n_frames=10, reported_count=0))
    assert values(fx.extract_uniform("x.mp4", 4)) == [0, 3, 6, 9]


def test_sequential_fallback_returns_all_when_video_is_short(use_cap):
    use_cap(FakeCap(n_frames=3, reported_count=0))
    assert values(fx.extract_uniform("x.mp4", 16)) == [0, 1, 2]


def test_sequential_fallback_on_empty_stream_raises(use_cap):
    use_cap(FakeCap(n_frames=0, reported_count=0))
    with pytest.raises(EmptyVideoError):
        fx.extract_uniform("x.mp4", 4)


def test_undecodable_frames_are_skipped(use_cap):
    # The container claims 5 frames but only 3 decode: the readable ones are still returned.
    use_cap(FakeCap(n_frames=3, reported_count=5))
    assert values(fx.extract_uniform("x.mp4", 5)) == [0, 1, 2]


def test_no_decodable_frames_raises(use_cap):
    use_cap(FakeCap(n_frames=0, reported_count=5))
    with pytest.raises(EmptyVideoError):
        fx.extract_uniform("x.mp4", 5)


def test_by_fps_stops_when_decoding_ends(use_cap):
    use_cap(FakeCap(n_frames=5, reported_count=0, fps=25.0))
    assert values(fx.extract_by_fps("x.mp4", target_fps=25.0, max_frames=None)) == [0, 1, 2, 3, 4]


def test_by_fps_steps_over_frames(use_cap):
    use_cap(FakeCap(n_frames=50, reported_count=50, fps=25.0))
    assert values(fx.extract_by_fps("x.mp4", target_fps=5.0, max_frames=None)) == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]


def test_by_fps_on_empty_stream_raises(use_cap):
    use_cap(FakeCap(n_frames=0, reported_count=0))
    with pytest.raises(EmptyVideoError):
        fx.extract_by_fps("x.mp4", target_fps=1.0)
