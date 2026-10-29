import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, videos
from app.core.config import settings
from app.models.registry import get_predictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the model once at startup: a bad FORGERY_MODEL_PATH fails fast instead of on the first request.
    predictor = get_predictor()
    logger.info("Predictor ready: %s (threshold=%s, aggregation=%s)", predictor.name,
                settings.decision_threshold, settings.aggregation)
    yield

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    description="Backend/pipeline service for AI-generated video forgery detection (SKIT CSE final-year project).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(videos.router)
