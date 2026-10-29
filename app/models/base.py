"""Model interface: the contract between the backend and the ML team's CNN.

Rishita's trained model only needs to subclass `BasePredictor` and implement `predict`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from app.core.config import settings

REAL = "REAL"
FORGED = "AI_GENERATED_FORGED"


@dataclass
class Prediction:
    label: str                 # REAL or AI_GENERATED_FORGED
    forged_probability: float  # 0..1, video-level
    frame_scores: list[float]  # per-frame forged probability


def aggregate_scores(scores: np.ndarray, method: str = "mean", topk_fraction: float = 0.3) -> float:
    """Collapse per-frame forged probabilities into one video-level probability.

    mean   - what the baseline training script evaluates (keep this unless you re-evaluate)
    median - robust to a few mis-scored frames
    topk   - mean of the highest `topk_fraction` of frames; helps when only part of a video is manipulated
    """
    if scores.size == 0:
        raise ValueError("scores must not be empty")
    if method == "mean":
        return float(scores.mean())
    if method == "median":
        return float(np.median(scores))
    if method == "topk":
        k = max(1, int(np.ceil(scores.size * topk_fraction)))
        return float(np.sort(scores)[-k:].mean())
    raise ValueError(f"Unknown aggregation method: {method!r}")


class BasePredictor(ABC):
    name: str = "base"

    @abstractmethod
    def predict_frames(self, batch: np.ndarray) -> np.ndarray:
        """Take a float32 batch (N, H, W, 3) and return per-frame forged probabilities, shape (N,)."""

    def predict(
        self,
        batch: np.ndarray,
        threshold: float | None = None,
        aggregation: str | None = None,
        topk_fraction: float | None = None,
    ) -> Prediction:
        threshold = settings.decision_threshold if threshold is None else threshold
        scores = np.clip(np.asarray(self.predict_frames(batch), dtype=np.float32), 0.0, 1.0)
        prob = aggregate_scores(
            scores,
            aggregation or settings.aggregation,
            settings.topk_fraction if topk_fraction is None else topk_fraction,
        )
        return Prediction(
            label=FORGED if prob >= threshold else REAL,
            forged_probability=prob,
            frame_scores=[float(s) for s in scores],
        )
