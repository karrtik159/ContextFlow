"""
Request correlation — one id per request, on the response and on every log line.

Phase 6. Before this, nothing tied a log line to its request: logs identified
work by `query[:80]`, and a background MemoryCrew failure was uncorrelatable
with the request that spawned it.

Three pieces:

- `RequestIDMiddleware` — honours an inbound `X-Request-ID` (sanitized: the
  header is caller-controlled and goes into logs) or mints one, stores it in a
  ContextVar for the request's lifetime, and returns it on the response.
- `get_request_id()` — read anywhere, including code that has no Request.
  Background tasks scheduled DURING a request inherit the ContextVar snapshot,
  so MemoryCrew logs carry the id of the request that spawned them.
- `install_record_factory()` — stamps `request_id` onto every LogRecord, so a
  formatter can opt in with `%(request_id)s` and existing formatters are
  untouched. Idempotent.
"""

from __future__ import annotations

import logging
import re
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

REQUEST_ID_HEADER = "X-Request-ID"

# No id — e.g. startup, scripts, tests that never pass through the middleware.
_NO_REQUEST = "-"

_request_id_var: ContextVar[str] = ContextVar("request_id", default=_NO_REQUEST)

# Caller-supplied ids are untrusted input headed for log files: strip anything
# that could fake log structure, cap the length.
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_ID_LEN = 64


def get_request_id() -> str:
    return _request_id_var.get()


def _sanitize(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = _SAFE_ID_RE.sub("", raw)[:_MAX_ID_LEN]
    return cleaned or None


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = _sanitize(request.headers.get(REQUEST_ID_HEADER)) or uuid.uuid4().hex[:16]
        token = _request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            _request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


_factory_installed = False


def install_record_factory() -> None:
    """Stamp request_id onto every LogRecord. Safe to call more than once."""
    global _factory_installed
    if _factory_installed:
        return
    _factory_installed = True

    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        record.request_id = get_request_id()
        return record

    logging.setLogRecordFactory(factory)
