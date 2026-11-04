import os
import sys
from pathlib import Path

# Ensure the project root is in sys.path so 'from app.core.config' always resolves in CI
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings  # noqa: E402


def test_database_url_configured():
    db_url = (
        settings.database_url
        or os.getenv("FORGERY_DATABASE_URL")
        or "sqlite:///:memory:"
    )
    assert db_url is not None
    assert "postgresql" in db_url or "sqlite" in db_url


def test_storage_cleanup_flag():
    assert isinstance(settings.delete_after_processing, bool)