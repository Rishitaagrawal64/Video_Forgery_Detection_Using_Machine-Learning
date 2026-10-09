"""PLACEHOLDER predictor so the whole pipeline runs end-to-end before the real CNN exists.

It is NOT a forgery detector. It maps image sharpness (Laplacian variance) to a score purely
to produce deterministic, input-dependent output for integration testing. Replace it with the
trained CNN (see registry.py) once the ML team delivers the model.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.models.base import BasePredictor


class HeuristicPredictor(BasePredictor):
    name = "heuristic-placeholder"

    def predict_frames(self, batch: np.ndarray) -> np.ndarray:
        scores = []
        for x in batch:
            gray = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            # smoother (lower variance) frames -> higher "forged" score
            scores.append(1.0 / (1.0 + sharp / 100.0))
        return np.array(scores, dtype=np.float32)
