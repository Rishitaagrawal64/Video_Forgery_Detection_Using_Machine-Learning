"""Thread-safe in-memory job store.

Deliberately behind a small interface so Tapesh's database layer (PostgreSQL/MongoDB/Firebase)
can replace it later without touching the API routes.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, video_id: str, filename: str, path: str) -> dict[str, Any]:
        with self._lock:
            self._jobs[video_id] = {
                "video_id": video_id,
                "filename": filename,
                "path": path,
                "status": "uploaded",
                "created_at": datetime.now(timezone.utc),
                "result": None,
                "error": None,
            }
            return dict(self._jobs[video_id])

    def get(self, video_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(video_id)
            return dict(job) if job else None

    def update(self, video_id: str, **fields: Any) -> None:
        with self._lock:
            if video_id in self._jobs:
                self._jobs[video_id].update(fields)

    def list(self, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)
            rows = rows[offset:] if limit is None else rows[offset : offset + limit]
            return [dict(j) for j in rows]

    def delete(self, video_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(video_id, None) is not None

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex


def _build_store():
    """PostgreSQL/SQLite when FORGERY_DATABASE_URL is set, otherwise the in-memory store."""
    from app.core.config import settings

    if settings.database_url:
        from app.services.db_job_store import SqlJobStore

        return SqlJobStore(settings.database_url)
    return JobStore()


job_store = _build_store()
