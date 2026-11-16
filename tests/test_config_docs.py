"""docs/configuration.md must stay in sync with app/core/config.py."""
from pathlib import Path

import pytest

from app.core.config import Settings

DOC = (Path(__file__).resolve().parents[1] / "docs" / "configuration.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("field", list(Settings.model_fields))
def test_every_setting_is_documented(field):
    assert f"`FORGERY_{field.upper()}`" in DOC, f"add FORGERY_{field.upper()} to docs/configuration.md"


def test_documented_list_syntax_works(monkeypatch):
    monkeypatch.setenv("FORGERY_ALLOWED_EXTENSIONS", '[".mp4", ".mov"]')
    monkeypatch.setenv("FORGERY_CORS_ORIGINS", '["https://example.org"]')
    s = Settings(_env_file=None)
    assert s.allowed_extensions == (".mp4", ".mov")
    assert s.cors_origins == ["https://example.org"]
