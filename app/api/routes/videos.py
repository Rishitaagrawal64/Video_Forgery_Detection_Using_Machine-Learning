from __future__ import annotations

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile, status

from app.core.config import settings
from app.core.exceptions import VideoError
from app.models.registry import get_predictor
from app.schemas.video import (
    AnalysisResult,
    AnalyzeRequest,
    JobResponse,
    UploadResponse,
    VideoInfoOut,
)
from app.services.job_store import job_store
from app.services.pipeline import run_pipeline

router = APIRouter(prefix="/api/v1", tags=["videos"])


def _to_job_response(job: dict) -> JobResponse:
    return JobResponse(**{k: job[k] for k in ("video_id", "filename", "status", "created_at", "result", "error")})


def _process(video_id: str, num_frames: int, detect_faces: bool) -> None:
    job = job_store.get(video_id)
    if job is None:
        return
    job_store.update(video_id, status="processing")
    started = time.perf_counter()
    try:
        predictor = get_predictor()
        res = run_pipeline(job["path"], predictor, num_frames=num_frames, detect_faces=detect_faces)
        v = res.video
        result = AnalysisResult(
            label=res.prediction.label,
            forged_probability=res.prediction.forged_probability,
            frame_scores=res.prediction.frame_scores,
            frames_analyzed=res.num_frames,
            video=VideoInfoOut(
                width=v.width, height=v.height, fps=v.fps,
                frame_count=v.frame_count, duration_sec=v.duration_sec,
            ),
            model=predictor.name,
            processing_sec=round(time.perf_counter() - started, 3),
        )
        job_store.update(video_id, status="completed", result=result)
    except VideoError as exc:
        job_store.update(video_id, status="failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the job record
        job_store.update(video_id, status="failed", error=f"Internal error: {exc}")
    finally:
        if settings.delete_after_processing:  # NFR-005
            Path(job["path"]).unlink(missing_ok=True)
            shutil.rmtree(settings.frames_dir / Path(job["path"]).stem, ignore_errors=True)


@router.post("/videos", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_video(file: UploadFile = File(...)) -> UploadResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(415, f"Unsupported file type {ext!r}. Allowed: {', '.join(settings.allowed_extensions)}")

    video_id = job_store.new_id()
    dest = settings.upload_dir / f"{video_id}{ext}"
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)  # never leave partial uploads behind
        raise

    job_store.create(video_id, file.filename or dest.name, str(dest))
    return UploadResponse(video_id=video_id, filename=file.filename or dest.name, status="uploaded")


@router.post("/videos/{video_id}/analyze", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def analyze_video(video_id: str, background: BackgroundTasks, body: AnalyzeRequest | None = None) -> JobResponse:
    job = job_store.get(video_id)
    if job is None:
        raise HTTPException(404, "Video not found")
    if job["status"] == "processing":
        raise HTTPException(409, "Video is already being processed")
    if not Path(job["path"]).exists():
        raise HTTPException(410, "Video file was already deleted after processing; upload it again")
    body = body or AnalyzeRequest()
    job_store.update(video_id, status="processing", result=None, error=None)
    background.add_task(_process, video_id, body.num_frames, body.detect_faces)
    return _to_job_response(job_store.get(video_id))


@router.get("/jobs/{video_id}", response_model=JobResponse)
def get_job(video_id: str) -> JobResponse:
    job = job_store.get(video_id)
    if job is None:
        raise HTTPException(404, "Video not found")
    return _to_job_response(job)


@router.get("/jobs", response_model=list[JobResponse])
def list_jobs(limit: int = Query(20, ge=1, le=200), offset: int = Query(0, ge=0)) -> list[JobResponse]:
    """History, newest first (FR-005)."""
    return [_to_job_response(j) for j in job_store.list(limit=limit, offset=offset)]


@router.delete("/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_video(video_id: str) -> None:
    job = job_store.get(video_id)
    if job is None:
        raise HTTPException(404, "Video not found")
    Path(job["path"]).unlink(missing_ok=True)
    shutil.rmtree(settings.frames_dir / Path(job["path"]).stem, ignore_errors=True)
    job_store.delete(video_id)
