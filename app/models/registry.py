"""Predictor registry. Add the real CNN here when it is ready."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import settings
from app.models.base import BasePredictor
from app.models.heuristic import HeuristicPredictor


class KerasPredictor(BasePredictor):
    """Example adapter for a saved Keras model (.keras / .h5). Requires `pip install tensorflow`.

    Expects a model that maps (N, H, W, 3) -> (N, 1) or (N,) forged probabilities.
    """

    name = "keras-cnn"

    def __init__(self, model_path: str):
        import tensorflow as tf  # imported lazily so TF is optional

        self.model = tf.keras.models.load_model(model_path)

    def predict_frames(self, batch):
        return self.model.predict(batch, verbose=0).reshape(len(batch))


class TorchPredictor(BasePredictor):
    """Adapter for a TorchScript model (`torch.jit.save`). Requires `pip install torch`.

    Receives channels-last numpy batches and feeds (N, 3, H, W); model must output (N,) or (N, 1)
    logits (sigmoid is applied here) -- change if your model already outputs probabilities.
    """

    name = "torch-cnn"

    def __init__(self, model_path: str):
        import torch

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(model_path, map_location=self.device).eval()

    def predict_frames(self, batch):
        torch = self._torch
        x = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(self.device)
        with torch.inference_mode():
            return torch.sigmoid(self.model(x).reshape(len(batch))).cpu().numpy()


def _require_model_path(kind: str) -> str:
    if not settings.model_path:
        raise RuntimeError(f"Set FORGERY_MODEL_PATH to use the {kind} predictor")
    if not Path(settings.model_path).exists():
        raise FileNotFoundError(
            f"Model file not found: {settings.model_path!r} (FORGERY_MODEL_PATH). "
            "Check the path, or train one with ml/train_baseline.py."
        )
    return settings.model_path


def _build_torch() -> BasePredictor:
    return TorchPredictor(_require_model_path("torch"))


def _build_keras() -> BasePredictor:
    return KerasPredictor(_require_model_path("keras"))


_FACTORIES = {
    "heuristic": HeuristicPredictor,
    "keras": _build_keras,
    "torch": _build_torch,
}


@lru_cache(maxsize=1)
def get_predictor() -> BasePredictor:
    try:
        return _FACTORIES[settings.predictor]()
    except KeyError:
        raise RuntimeError(f"Unknown predictor {settings.predictor!r}; options: {list(_FACTORIES)}")
