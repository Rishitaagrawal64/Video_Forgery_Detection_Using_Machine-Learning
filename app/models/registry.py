"""Predictor registry. Add the real CNN here when it is ready."""
from __future__ import annotations

from functools import lru_cache

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


def _build_keras() -> BasePredictor:
    if not settings.model_path:
        raise RuntimeError("Set FORGERY_MODEL_PATH to use the keras predictor")
    return KerasPredictor(settings.model_path)


_FACTORIES = {
    "heuristic": HeuristicPredictor,
    "keras": _build_keras,
}


@lru_cache(maxsize=1)
def get_predictor() -> BasePredictor:
    try:
        return _FACTORIES[settings.predictor]()
    except KeyError:
        raise RuntimeError(f"Unknown predictor {settings.predictor!r}; options: {list(_FACTORIES)}")
