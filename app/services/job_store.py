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

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(j) for j in self._jobs.values()]

    def delete(self, video_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(video_id, None) is not None

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex


job_store = JobStore()
