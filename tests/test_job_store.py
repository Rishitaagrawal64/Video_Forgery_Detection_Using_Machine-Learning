from datetime import UTC, datetime, timedelta

import pytest

from app.services import job_store as js


class TickingClock:
    """Each call is one second later, so ordering never depends on the real clock's resolution."""

    def __init__(self):
        self.t = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self, tz=None):
        self.t += timedelta(seconds=1)
        return self.t


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(js, "datetime", TickingClock())
    s = js.JobStore()
    for i in range(5):
        s.create(f"id{i}", f"clip{i}.mp4", f"/tmp/clip{i}.mp4")
    return s


def test_new_jobs_start_as_uploaded(store):
    job = store.get("id0")
    assert job["status"] == "uploaded" and job["result"] is None and job["error"] is None


def test_get_returns_a_copy(store):
    store.get("id0")["status"] = "completed"
    assert store.get("id0")["status"] == "uploaded"


def test_get_and_delete_unknown_id(store):
    assert store.get("nope") is None
    assert store.delete("nope") is False


def test_update_changes_fields_and_ignores_unknown_ids(store):
    store.update("id1", status="failed", error="boom")
    assert store.get("id1")["status"] == "failed" and store.get("id1")["error"] == "boom"
    store.update("nope", status="failed")  # must not raise or create anything
    assert store.get("nope") is None


def test_list_is_newest_first(store):
    assert [j["video_id"] for j in store.list()] == ["id4", "id3", "id2", "id1", "id0"]


def test_list_pagination(store):
    assert [j["video_id"] for j in store.list(limit=2)] == ["id4", "id3"]
    assert [j["video_id"] for j in store.list(limit=2, offset=2)] == ["id2", "id1"]
    assert [j["video_id"] for j in store.list(offset=4)] == ["id0"]
    assert store.list(limit=2, offset=10) == []


def test_delete_removes_the_job(store):
    assert store.delete("id2") is True
    assert store.get("id2") is None and len(store.list()) == 4


def test_new_ids_are_unique_hex():
    ids = {js.JobStore.new_id() for _ in range(100)}
    assert len(ids) == 100 and all(len(i) == 32 for i in ids)
