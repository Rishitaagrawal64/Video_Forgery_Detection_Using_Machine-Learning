from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

JobStatus = Literal["uploaded", "processing", "completed", "failed"]


class VideoInfoOut(BaseModel):
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


class AnalysisResult(BaseModel):
    label: Literal["REAL", "AI_GENERATED_FORGED"]
    forged_probability: float = Field(ge=0, le=1)
    frame_scores: list[float]
    frames_analyzed: int
    video: VideoInfoOut
    model: str


class UploadResponse(BaseModel):
    video_id: str
    filename: str
    status: JobStatus


class JobResponse(BaseModel):
    video_id: str
    filename: str
    status: JobStatus
    created_at: datetime
    result: AnalysisResult | None = None
    error: str | None = None


class AnalyzeRequest(BaseModel):
    num_frames: int = Field(default=16, ge=1, le=128)
    detect_faces: bool = False


class HealthResponse(BaseModel):
    status: str
    version: str
    predictor: str
