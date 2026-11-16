"""Per-request id and timing: every response carries X-Request-ID and X-Process-Time, and one access line is logged."""
from __future__ import annotations

import logging
import re
import time
import uuid

from fastapi import FastAPI, Request

logger = logging.getLogger("app.access")

# Accept a client-supplied id only if it is short and plain, so it cannot inject text into the logs.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def add_request_context(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request_id = supplied if _SAFE_ID.match(supplied) else uuid.uuid4().hex[:12]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed_ms:.1f}ms"
        logger.info(
            "%s %s -> %s in %.1f ms [%s]",
            request.method, request.url.path, response.status_code, elapsed_ms, request_id,
        )
        return response
