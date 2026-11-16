"""Bad FORGERY_* values must fail at startup with a clear error, not midway through a job."""
import pytest
from pydantic import ValidationError

from app.core.config import Settings


def make(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_defaults_are_valid():
    s = make()
    assert s.normalize_mode == "unit" and s.num_frames == 16 and s.frame_size == 224


@pytest.mark.parametrize("mode", ["unit", "imagenet", "tanh"])
def test_known_normalize_modes(mode):
    assert make(normalize_mode=mode).normalize_mode == mode


@pytest.mark.parametrize(
    "bad",
    [
        {"normalize_mode": "image_net"},
        {"num_frames": 0},
        {"num_frames": 500},
        {"frame_size": 4},
        {"max_upload_mb": 0},
        {"decision_threshold": 1.5},
        {"stale_file_ttl_hours": -1},
        {"aggregation": "max"},
    ],
)
def test_invalid_values_are_rejected(bad):
    with pytest.raises(ValidationError):
        make(**bad)


def test_log_level_is_validated_and_normalized():
    assert make(log_level="debug").log_level == "DEBUG"
    with pytest.raises(ValidationError, match="FORGERY_LOG_LEVEL"):
        make(log_level="loud")


def test_env_vars_are_read(monkeypatch):
    monkeypatch.setenv("FORGERY_NORMALIZE_MODE", "imagenet")
    assert make().normalize_mode == "imagenet"
    monkeypatch.setenv("FORGERY_NORMALIZE_MODE", "wrong")
    with pytest.raises(ValidationError):
        make()
