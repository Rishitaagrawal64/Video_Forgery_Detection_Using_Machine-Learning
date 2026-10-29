from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "AI Video Forgery Detection API"
    app_version: str = "0.1.0"

    upload_dir: Path = BASE_DIR / "data" / "uploads"
    frames_dir: Path = BASE_DIR / "data" / "frames"

    max_upload_mb: int = 50  # Functional Requirements, Constraint 2
    delete_after_processing: bool = True  # NFR-005: privacy
    allowed_extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    # Frame extraction / preprocessing defaults
    num_frames: int = 16
    frame_size: int = 224
    normalize_mode: str = "unit"  # "unit" -> [0,1], "imagenet", "tanh" -> [-1,1]
    save_frames: bool = False
    detect_faces: bool = False

    # Which predictor to load (see app/models/registry.py)
    predictor: str = "heuristic"
    model_path: str | None = None

    # Video-level decision. The trained model's validation-selected threshold is written to
    # ml/out/metrics.json ("threshold") by ml/train_baseline.py - set FORGERY_DECISION_THRESHOLD to it.
    decision_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    aggregation: Literal["mean", "median", "topk"] = "mean"  # how frame scores become one video score
    topk_fraction: float = Field(default=0.3, gt=0.0, le=1.0)  # used when aggregation == "topk"

    # e.g. postgresql+psycopg://user:pass@db:5432/forgery  (empty -> in-memory store, jobs lost on restart)
    database_url: str | None = None

    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    model_config = SettingsConfigDict(env_prefix="FORGERY_", env_file=".env", extra="ignore")


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.frames_dir.mkdir(parents=True, exist_ok=True)
