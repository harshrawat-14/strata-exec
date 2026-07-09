"""
web/observability.py

Structured logging and request tracing for the FastAPI backend.

StructuredLogger emits single-line, greppable, key=value structured log
records for the events that matter operationally (simulation lifecycle,
Rust subprocess failures, demo-mode cache hits/misses) instead of free-form
strings. RequestIDMiddleware stamps every request/response with a
correlation ID so a single request can be traced across log lines.

Does not touch demo_middleware.py, routes, or any existing middleware —
purely additive.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("strataexec")


class StructuredLogger:
    """
    Thin wrapper emitting structured (key=value) log lines for operationally
    significant events. Each method logs one event type at INFO (or ERROR
    for failures) so events are greppable by name, e.g.:

        grep "event=simulation_complete" app.log
    """

    def __init__(self, log: logging.Logger = logger):
        self._log = log

    @staticmethod
    def _fmt(event: str, **fields: Any) -> str:
        parts = [f"event={event}"]
        parts += [f"{k}={v}" for k, v in fields.items()]
        return " ".join(parts)

    def simulation_start(self, request_id: str, price_model: str, **params: Any) -> None:
        self._log.info(self._fmt(
            "simulation_start", request_id=request_id, price_model=price_model, **params,
        ))

    def simulation_complete(self, request_id: str, duration_ms: float, **result_meta: Any) -> None:
        self._log.info(self._fmt(
            "simulation_complete", request_id=request_id, duration_ms=round(duration_ms, 2), **result_meta,
        ))

    def rust_process_error(self, request_id: str, binary: str, error: str) -> None:
        self._log.error(self._fmt(
            "rust_process_error", request_id=request_id, binary=binary, error=error,
        ))

    def demo_cache_hit(self, request_id: str, path: str, preset_key: str) -> None:
        self._log.info(self._fmt(
            "demo_cache_hit", request_id=request_id, path=path, preset_key=preset_key,
        ))

    def demo_cache_miss(self, request_id: str, path: str, preset_key: str) -> None:
        self._log.info(self._fmt(
            "demo_cache_miss", request_id=request_id, path=path, preset_key=preset_key,
        ))


structured_logger = StructuredLogger()


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Assigns a UUID4 correlation ID to every request (or reuses an incoming
    X-Request-ID header from an upstream proxy), stores it on
    request.state.request_id, and echoes it back as a response header so
    a client-reported issue can be traced to server-side log lines.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        logger.info(
            StructuredLogger._fmt(
                "request_complete",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
        )
        return response
