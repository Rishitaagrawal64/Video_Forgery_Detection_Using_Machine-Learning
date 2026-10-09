import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> str:
    """Synthetic 3-second 25fps 160x120 video with moving shapes and noise."""
    path = tmp_path_factory.mktemp("vid") / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (160, 120))
    rng = np.random.default_rng(0)
    for i in range(75):
        frame = rng.integers(0, 60, (120, 160, 3), dtype=np.uint8)
        cv2.circle(frame, (10 + i * 2, 60), 20, (0, 200, 255), -1)
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    """Keep test uploads/frames out of the real data/ folders."""
    from app.core.config import settings

    up, fr = tmp_path / "uploads", tmp_path / "frames"
    up.mkdir()
    fr.mkdir()
    monkeypatch.setattr(settings, "upload_dir", up)
    monkeypatch.setattr(settings, "frames_dir", fr)
