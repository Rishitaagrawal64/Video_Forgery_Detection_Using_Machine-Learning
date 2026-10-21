import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def _upload(path: str) -> str:
    with open(path, "rb") as f:
        r = client.post("/api/v1/videos", files={"file": ("sample.avi", f, "video/x-msvideo")})
    assert r.status_code == 201
    return r.json()["video_id"]


def test_upload_analyze_flow(sample_video):
    vid = _upload(sample_video)
    r = client.post(f"/api/v1/videos/{vid}/analyze", json={"num_frames": 8})
    assert r.status_code == 202

    for _ in range(50):
        job = client.get(f"/api/v1/jobs/{vid}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert job["status"] == "completed", job
    res = job["result"]
    assert res["label"] in ("REAL", "AI_GENERATED_FORGED")
    assert 0 <= res["forged_probability"] <= 1
    assert len(res["frame_scores"]) == res["frames_analyzed"] == 8

    assert client.delete(f"/api/v1/videos/{vid}").status_code == 204
    assert client.get(f"/api/v1/jobs/{vid}").status_code == 404


def test_rejects_bad_extension():
    r = client.post("/api/v1/videos", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415

def test_rejects_empty_upload():
    from app.core.config import settings

    r = client.post(
        "/api/v1/videos",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )

    assert r.status_code == 400
    assert r.json()["detail"] == "Uploaded file is empty"
    assert list(settings.upload_dir.iterdir()) == []

def test_corrupt_video_marks_job_failed(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a real video")
    vid = _upload_bytes(bad)
    client.post(f"/api/v1/videos/{vid}/analyze")
    for _ in range(50):
        job = client.get(f"/api/v1/jobs/{vid}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "failed"
    assert job["error"]


def _upload_bytes(path) -> str:
    with open(path, "rb") as f:
        r = client.post("/api/v1/videos", files={"file": (path.name, f, "video/mp4")})
    return r.json()["video_id"]


@pytest.mark.parametrize("path", ["/api/v1/jobs/unknown"])
def test_unknown_job_404(path):
    assert client.get(path).status_code == 404


def test_upload_over_limit_rejected_and_cleaned(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_upload_mb", 1)
    r = client.post("/api/v1/videos", files={"file": ("big.mp4", b"0" * (2 * 1024 * 1024), "video/mp4")})
    assert r.status_code == 413
    assert list(settings.upload_dir.iterdir()) == []


def test_file_deleted_after_processing(sample_video):
    from app.core.config import settings

    vid = _upload(sample_video)
    client.post(f"/api/v1/videos/{vid}/analyze", json={"num_frames": 4})
    for _ in range(50):
        job = client.get(f"/api/v1/jobs/{vid}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "completed"
    assert job["result"]["processing_sec"] is not None
    assert list(settings.upload_dir.iterdir()) == []  # NFR-005
    assert client.post(f"/api/v1/videos/{vid}/analyze").status_code == 410
