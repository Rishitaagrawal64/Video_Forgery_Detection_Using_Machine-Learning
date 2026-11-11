"""Smoke test for a running stack (docker compose up): UI, API health, and database access through the proxy.

Usage: python scripts/smoke_test.py [ui_url] [api_url]     defaults: http://localhost:3000  http://localhost:8000
"""
import json
import sys
import time
import urllib.request


def fetch(url: str, retries: int = 15, delay: float = 2.0) -> tuple[int, str]:
    last: Exception | None = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001 - keep retrying while containers start
            last = exc
            time.sleep(delay)
    raise SystemExit(f"FAIL {url}: {last}")


def main() -> None:
    ui = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:3000"
    api = sys.argv[2].rstrip("/") if len(sys.argv) > 2 else "http://localhost:8000"

    status, _ = fetch(ui + "/")
    assert status == 200, f"UI returned {status}"
    print("ok   UI is served")

    status, body = fetch(api + "/health")
    assert status == 200 and json.loads(body)["status"] == "ok", body
    print("ok   API /health")

    # goes UI nginx -> API -> PostgreSQL, so it proves the proxy and the database connection
    status, body = fetch(ui + "/api/v1/jobs?limit=1")
    assert status == 200 and isinstance(json.loads(body), list), body
    print("ok   /api/v1/jobs through the UI proxy (database reachable)")


if __name__ == "__main__":
    main()
