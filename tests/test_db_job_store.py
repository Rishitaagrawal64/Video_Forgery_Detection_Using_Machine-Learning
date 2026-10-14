import time

import pytest
from fastapi.testclient import TestClient

from app.services.db_job_store import SqlJobStore

RESULT = {
    "label": "REAL", "forged_probability": 0.2, "frame_scores": [0.1, 0.3], "frames_analyzed": 2,
    "video": {"width": 1, "height": 1, "fps": 25.0, "frame_count": 2, "duration_sec": 0.1},
    "model": "test", "processing_sec": 0.5,
}


@pytest.fixture
def store(tmp_path):
    return SqlJobStore(f"sqlite:///{tmp_path / 'test.db'}")


def test_create_get_and_missing(store):
    job = store.create("v1", "a.mp4", "/tmp/a.mp4")
    assert job["status"] == "uploaded" and job["result"] is None
    assert store.get("v1")["filename"] == "a.mp4"
    assert store.get("nope") is None


def test_full_lifecycle_and_result_roundtrip(store):
    store.create("v1", "a.mp4", "/tmp/a.mp4")
    store.update("v1", status="processing", result=None, error=None)
    store.update("v1", status="processing")  # route + background task both set it
    store.update("v1", status="completed", result=RESULT)
    job = store.get("v1")
    assert job["status"] == "completed"
    assert job["result"] == RESULT
    assert job["created_at"].tzinfo is not None


def test_failed_keeps_error(store):
    store.create("v1", "a.mp4", "/tmp/a.mp4")
    store.update("v1", status="processing")
    store.update("v1", status="failed", error="boom")
    job = store.get("v1")
    assert job["status"] == "failed" and job["error"] == "boom" and job["result"] is None


def test_reanalyze_adds_history_row(store):
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    from app.services.db_job_store import PredictionRow

    store.create("v1", "a.mp4", "/tmp/a.mp4")
    store.update("v1", status="processing")
    store.update("v1", status="completed", result=RESULT)
    store.update("v1", status="processing", result=None, error=None)  # second run
    assert store.get("v1")["status"] == "processing" and store.get("v1")["result"] is None
    with Session(store.engine) as s:
        assert s.scalar(select(func.count()).select_from(PredictionRow)) == 2


def test_list_paging_newest_first_and_delete(store):
    for i in range(5):
        store.create(f"v{i}", f"{i}.mp4", f"/tmp/{i}.mp4")
        time.sleep(0.01)
    ids = [j["video_id"] for j in store.list()]
    assert ids == ["v4", "v3", "v2", "v1", "v0"]
    assert [j["video_id"] for j in store.list(limit=2, offset=1)] == ["v3", "v2"]
    assert store.delete("v2") is True and store.delete("v2") is False
    assert store.get("v2") is None and len(store.list()) == 4


def test_history_survives_restart(tmp_path):
    url = f"sqlite:///{tmp_path / 'persist.db'}"
    a = SqlJobStore(url)
    a.create("v1", "a.mp4", "/tmp/a.mp4")
    a.update("v1", status="processing")
    a.update("v1", status="completed", result=RESULT)
    b = SqlJobStore(url)  # new object = like restarting the server
    assert b.get("v1")["result"] == RESULT


def test_api_with_database_store(monkeypatch, tmp_path, sample_video):
    import app.api.routes.videos as videos
    from app.main import app

    monkeypatch.setattr(videos, "job_store", SqlJobStore(f"sqlite:///{tmp_path / 'api.db'}"))
    client = TestClient(app)
    with open(sample_video, "rb") as f:
        vid = client.post("/api/v1/videos", files={"file": ("s.avi", f, "video/x-msvideo")}).json()["video_id"]
    assert client.post(f"/api/v1/videos/{vid}/analyze", json={"num_frames": 4}).status_code == 202
    for _ in range(50):
        job = client.get(f"/api/v1/jobs/{vid}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "completed", job
    hist = client.get("/api/v1/jobs?limit=10").json()
    assert [j["video_id"] for j in hist] == [vid]
    assert client.get("/api/v1/jobs?limit=0").status_code == 422
