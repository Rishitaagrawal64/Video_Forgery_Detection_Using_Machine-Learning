from fastapi import APIRouter

from app.core.config import settings
from app.models.registry import get_predictor
from app.schemas.video import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version, predictor=get_predictor().name)
