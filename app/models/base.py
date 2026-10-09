"""Model interface: the contract between the backend and the ML team's CNN.

Rishita's trained model only needs to subclass `BasePredictor` and implement `predict`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

REAL = "REAL"
FORGED = "AI_GENERATED_FORGED"


@dataclass
class Prediction:
    label: str                 # REAL or AI_GENERATED_FORGED
    forged_probability: float  # 0..1, video-level
    frame_scores: list[float]  # per-frame forged probability


class BasePredictor(ABC):
    name: str = "base"

    @abstractmethod
    def predict_frames(self, batch: np.ndarray) -> np.ndarray:
        """Take a float32 batch (N, H, W, 3) and return per-frame forged probabilities, shape (N,)."""

    def predict(self, batch: np.ndarray, threshold: float = 0.5) -> Prediction:
        scores = np.clip(np.asarray(self.predict_frames(batch), dtype=np.float32), 0.0, 1.0)
        prob = float(scores.mean())
        return Prediction(
            label=FORGED if prob >= threshold else REAL,
            forged_probability=prob,
            frame_scores=[float(s) for s in scores],
        )
