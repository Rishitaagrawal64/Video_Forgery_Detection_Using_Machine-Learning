"""Checks that the Docker setup and the database-URL handling agree with each other (FR-005, NFR-004, NFR-005)."""
from pathlib import Path

import yaml

from app.core.config import settings
from app.services.db_job_store import SqlJobStore
from app.services.job_store import JobStore, _build_store

ROOT = Path(__file__).resolve().parents[1]


def _compose_services() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def _env(service: dict) -> dict:
    env = service.get("environment", {})
    if isinstance(env, list):  # "KEY=value" list form
        return dict(item.split("=", 1) for item in env)
    return env


def test_compose_db_service_is_postgres():
    assert _compose_services()["db"]["image"].startswith("postgres")


def test_compose_api_uses_postgres_on_db_service():
    api = _compose_services()["api"]
    url = str(_env(api)["FORGERY_DATABASE_URL"])
    assert url.startswith("postgresql+psycopg://")
    assert "@db:5432/" in url
    assert "db" in api["depends_on"]  # API must wait for the database


def test_empty_database_url_falls_back_to_memory_store(monkeypatch):
    # an empty FORGERY_DATABASE_URL (as set in tests/conftest.py) must not try to reach Postgres
    monkeypatch.setattr(settings, "database_url", "")
    assert type(_build_store()) is JobStore


def test_sqlite_database_url_uses_sql_store(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "database_url", "sqlite:///" + (tmp_path / "jobs.db").as_posix())
    assert isinstance(_build_store(), SqlJobStore)


def test_storage_cleanup_flag():
    assert isinstance(settings.delete_after_processing, bool)