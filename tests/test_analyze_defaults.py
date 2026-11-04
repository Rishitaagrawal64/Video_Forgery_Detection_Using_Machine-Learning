"""An empty analyze request must use the server settings, not hard-coded API defaults."""
import time

import pytest
from fastapi.testclient import TestClient

from app.api.routes import videos
from app.core.config import settings
from app.main import app

client = TestClient(app)


def _upload(path: str) -> str:
    with open(path, "rb") as f:
        r = client.post("/api/v1/videos", files={"file": ("sample.avi", f, "video/x-msvideo")})
    assert r.status_code == 201
    return r.json()["video_id"]


def _wait(vid: str) -> dict:
    for _ in range(100):
        job = client.get(f"/api/v1/jobs/{vid}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []
    real = videos.run_pipeline

    def spy(path, predictor, **kwargs):
        calls.append(kwargs)
        return real(path, predictor, **kwargs)

    monkeypatch.setattr(videos, "run_pipeline", spy)
    return calls


def test_empty_body_uses_settings(sample_video, captured, monkeypatch):
    monkeypatch.setattr(settings, "num_frames", 5)
    monkeypatch.setattr(settings, "detect_faces", True)
    vid = _upload(sample_video)
    client.post(f"/api/v1/videos/{vid}/analyze")
    job = _wait(vid)
    assert job["status"] == "completed", job
    assert job["result"]["frames_analyzed"] == 5
    assert captured[0] == {"num_frames": None, "detect_faces": None}


def test_request_values_override_settings(sample_video, captured, monkeypatch):
    monkeypatch.setattr(settings, "num_frames", 5)
    vid = _upload(sample_video)
    client.post(f"/api/v1/videos/{vid}/analyze", json={"num_frames": 3, "detect_faces": False})
    job = _wait(vid)
    assert job["result"]["frames_analyzed"] == 3
    assert captured[0] == {"num_frames": 3, "detect_faces": False}
