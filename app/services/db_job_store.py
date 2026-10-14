"""Database-backed job store (PostgreSQL in Docker, SQLite for local tests).

Same public methods as the in-memory `JobStore`, so the API routes don't change.
Two tables:
  videos       one row per uploaded video (metadata only - the video file itself is never stored)
  predictions  one row per analysis run of a video (history/audit log, FR-005)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

JsonType = JSON().with_variant(JSONB(), "postgresql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class VideoRow(Base):
    __tablename__ = "videos"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(Text)
    path: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PredictionRow(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20))
    label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    forged_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_scores: Mapped[list | None] = mapped_column(JsonType, nullable=True)
    frames_analyzed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    video_info: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    processing_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def _aware(dt: datetime | None) -> datetime | None:
    # SQLite drops tzinfo; PostgreSQL keeps it. Normalise so the API always returns UTC-aware values.
    return dt.replace(tzinfo=timezone.utc) if dt is not None and dt.tzinfo is None else dt


class SqlJobStore:
    def __init__(self, url: str) -> None:
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
        self.engine = create_engine(url, **kwargs)
        Base.metadata.create_all(self.engine)  # creates the tables on first start
        self._Session = sessionmaker(self.engine, expire_on_commit=False)

    # ---- helpers -------------------------------------------------------------------------
    @staticmethod
    def _latest(s: Session, video_id: str) -> PredictionRow | None:
        return s.scalars(
            select(PredictionRow).where(PredictionRow.video_id == video_id).order_by(PredictionRow.id.desc()).limit(1)
        ).first()

    @staticmethod
    def _to_dict(v: VideoRow, p: PredictionRow | None) -> dict[str, Any]:
        result = None
        if p is not None and p.status == "completed" and p.label is not None:
            result = {
                "label": p.label,
                "forged_probability": p.forged_probability,
                "frame_scores": p.frame_scores,
                "frames_analyzed": p.frames_analyzed,
                "video": p.video_info,
                "model": p.model_name,
                "processing_sec": p.processing_sec,
            }
        return {
            "video_id": v.id,
            "filename": v.filename,
            "path": v.path,
            "status": p.status if p else "uploaded",
            "created_at": _aware(v.created_at),
            "result": result,
            "error": p.error if p else None,
        }

    # ---- same interface as JobStore ------------------------------------------------------
    def create(self, video_id: str, filename: str, path: str) -> dict[str, Any]:
        with self._Session() as s, s.begin():
            v = VideoRow(id=video_id, filename=filename, path=path)
            s.add(v)
        return self.get(video_id)  # type: ignore[return-value]

    def get(self, video_id: str) -> dict[str, Any] | None:
        with self._Session() as s:
            v = s.get(VideoRow, video_id)
            return self._to_dict(v, self._latest(s, video_id)) if v else None

    def update(self, video_id: str, **fields: Any) -> None:
        with self._Session() as s, s.begin():
            if s.get(VideoRow, video_id) is None:
                return
            p = self._latest(s, video_id)
            status = fields.get("status")
            # a new analysis run = a new prediction row (keeps the history); repeated "processing" reuses it
            if p is None or (p.status != "processing" and status == "processing"):
                p = PredictionRow(video_id=video_id, status=status or "processing")
                s.add(p)
            if status:
                p.status = status
                if status in ("completed", "failed"):
                    p.finished_at = _now()
            if "error" in fields:
                p.error = fields["error"]
            if "result" in fields:
                r = fields["result"]
                if r is not None and hasattr(r, "model_dump"):  # pydantic AnalysisResult from the route
                    r = r.model_dump()
                if r is None:
                    p.label = p.forged_probability = p.frame_scores = p.frames_analyzed = None
                    p.video_info = p.model_name = p.processing_sec = None
                else:
                    p.label = r["label"]
                    p.forged_probability = r["forged_probability"]
                    p.frame_scores = list(r["frame_scores"])
                    p.frames_analyzed = r["frames_analyzed"]
                    p.video_info = dict(r["video"])
                    p.model_name = r["model"]
                    p.processing_sec = r.get("processing_sec")

    def list(self, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        with self._Session() as s:
            q = select(VideoRow).order_by(VideoRow.created_at.desc()).offset(offset)
            if limit is not None:
                q = q.limit(limit)
            return [self._to_dict(v, self._latest(s, v.id)) for v in s.scalars(q)]

    def delete(self, video_id: str) -> bool:
        with self._Session() as s, s.begin():
            v = s.get(VideoRow, video_id)
            if v is None:
                return False
            for p in s.scalars(select(PredictionRow).where(PredictionRow.video_id == video_id)):
                s.delete(p)
            s.delete(v)
            return True

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex
