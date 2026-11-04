from fastapi import APIRouter

from app.core.config import settings
from app.models.registry import get_predictor
from app.schemas.video import ModelInfoResponse

router = APIRouter(prefix="/api/v1", tags=["model"])


@router.get("/model", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Active predictor and preprocessing/decision settings, so clients and reviewers can see what produced a result."""
    return ModelInfoResponse(
        predictor=get_predictor().name,
        num_frames=settings.num_frames,
        frame_size=settings.frame_size,
        normalize_mode=settings.normalize_mode,
        detect_faces=settings.detect_faces,
        aggregation=settings.aggregation,
        topk_fraction=settings.topk_fraction,
        decision_threshold=settings.decision_threshold,
    )
