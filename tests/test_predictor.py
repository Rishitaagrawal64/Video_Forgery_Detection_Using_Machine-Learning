import numpy as np
import pytest

from app.core.config import settings
from app.models.base import FORGED, REAL, BasePredictor, aggregate_scores


class FixedPredictor(BasePredictor):
    name = "fixed"

    def __init__(self, scores):
        self._scores = np.asarray(scores, dtype=np.float32)

    def predict_frames(self, batch):
        return self._scores


def _batch(n):
    return np.zeros((n, 4, 4, 3), dtype=np.float32)


def test_aggregate_methods():
    s = np.array([0.1, 0.1, 0.1, 0.9], dtype=np.float32)
    assert aggregate_scores(s, "mean") == pytest.approx(0.3)
    assert aggregate_scores(s, "median") == pytest.approx(0.1)
    assert aggregate_scores(s, "topk", 0.25) == pytest.approx(0.9)


def test_aggregate_rejects_bad_input():
    with pytest.raises(ValueError):
        aggregate_scores(np.array([]), "mean")
    with pytest.raises(ValueError):
        aggregate_scores(np.array([0.5]), "max")


def test_topk_always_uses_at_least_one_frame():
    assert aggregate_scores(np.array([0.2, 0.8]), "topk", 0.01) == pytest.approx(0.8)


def test_threshold_is_configurable(monkeypatch):
    p = FixedPredictor([0.6, 0.6, 0.6, 0.6])
    monkeypatch.setattr(settings, "decision_threshold", 0.5)
    assert p.predict(_batch(4)).label == FORGED
    monkeypatch.setattr(settings, "decision_threshold", 0.7)
    assert p.predict(_batch(4)).label == REAL


def test_explicit_args_override_settings(monkeypatch):
    monkeypatch.setattr(settings, "aggregation", "mean")
    p = FixedPredictor([0.0, 0.0, 0.0, 1.0])
    assert p.predict(_batch(4), threshold=0.5).label == REAL
    assert p.predict(_batch(4), threshold=0.5, aggregation="topk", topk_fraction=0.25).label == FORGED


def test_scores_are_clipped():
    pred = FixedPredictor([-1.0, 2.0]).predict(_batch(2))
    assert pred.frame_scores == [0.0, 1.0]
