import pytest

from app.core.config import settings
from app.models import registry
from app.models.heuristic import HeuristicPredictor


@pytest.fixture(autouse=True)
def fresh_cache():
    registry.get_predictor.cache_clear()
    yield
    registry.get_predictor.cache_clear()


def test_default_is_heuristic(monkeypatch):
    monkeypatch.setattr(settings, "predictor", "heuristic")
    assert isinstance(registry.get_predictor(), HeuristicPredictor)


def test_predictor_is_cached(monkeypatch):
    monkeypatch.setattr(settings, "predictor", "heuristic")
    assert registry.get_predictor() is registry.get_predictor()


def test_unknown_predictor_lists_options(monkeypatch):
    monkeypatch.setattr(settings, "predictor", "nope")
    with pytest.raises(RuntimeError, match="Unknown predictor.*heuristic"):
        registry.get_predictor()


@pytest.mark.parametrize("name", ["torch", "keras"])
def test_model_backed_predictors_require_a_path(monkeypatch, name):
    monkeypatch.setattr(settings, "predictor", name)
    monkeypatch.setattr(settings, "model_path", None)
    with pytest.raises(RuntimeError, match="FORGERY_MODEL_PATH"):
        registry.get_predictor()
