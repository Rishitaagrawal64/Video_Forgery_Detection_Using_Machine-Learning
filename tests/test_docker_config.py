import os
from app.core.config import settings

def test_database_url_configured():
    db_url = settings.database_url or os.getenv("FORGERY_DATABASE_URL", "sqlite:///:memory:")
    assert db_url is not None
    assert "postgresql" in db_url or "sqlite" in db_url

def test_storage_cleanup_flag():
    assert isinstance(settings.delete_after_processing, bool)