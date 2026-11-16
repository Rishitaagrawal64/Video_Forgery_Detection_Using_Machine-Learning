import logging
import re

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_response_has_request_id_and_timing():
    r = client.get("/health")
    assert re.fullmatch(r"[0-9a-f]{12}", r.headers["X-Request-ID"])
    assert re.fullmatch(r"\d+\.\d+ms", r.headers["X-Process-Time"])


def test_ids_are_unique_per_request():
    assert client.get("/health").headers["X-Request-ID"] != client.get("/health").headers["X-Request-ID"]


def test_safe_client_id_is_echoed():
    r = client.get("/health", headers={"X-Request-ID": "trace-abc_123.4"})
    assert r.headers["X-Request-ID"] == "trace-abc_123.4"


def test_unsafe_client_id_is_replaced():
    r = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
    assert r.headers["X-Request-ID"] != "bad id with spaces"
    assert re.fullmatch(r"[0-9a-f]{12}", r.headers["X-Request-ID"])


def test_error_responses_are_tagged_and_logged(caplog):
    with caplog.at_level(logging.INFO, logger="app.access"):
        r = client.get("/api/v1/jobs/does-not-exist")
    assert r.status_code == 404 and "X-Request-ID" in r.headers
    assert any("404" in rec.getMessage() and r.headers["X-Request-ID"] in rec.getMessage() for rec in caplog.records)
