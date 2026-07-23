"""
Per-scope rate limiting for the expensive endpoints.

Phase 6. Until now there was NO rate limiting anywhere — the Redis-backed
dependency in the boilerplate is a pass-through stub with no call sites — so
an authenticated user could burn unbounded LLM and embedding spend.

This is a sliding-window limiter held in process memory, keyed by resolved
tenant scope, applied AFTER authentication (the anonymous 401 happens first;
what this bounds is authenticated spend). Applied at the two spend-bearing
write paths: /rag/query and document ingest/replace.

IN-PROCESS ON PURPOSE. A single uvicorn process is the only deployment shape
this repo has (docker compose dev profile); a distributed limiter needs the
Redis that Phase 6 explicitly does not introduce. If the deployment grows
replicas, this module is the seam — same `check_rate_limit` call, backed by
Redis INCR/EXPIRE instead of a dict. The limit is per-process until then.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.metrics import increment

_lock = threading.Lock()
_windows: dict[str, deque[float]] = {}

_WINDOW_SECONDS = 60.0

# A dict that grows one deque per scope forever is a slow leak; prune scopes
# whose whole window has expired whenever the table gets large.
_PRUNE_THRESHOLD = 10_000


def check_rate_limit(key: str, *, limit: int | None = None) -> None:
    """Record one request for `key`; raise 429 when the window is over budget.

    `limit` defaults to RATE_LIMIT_RPM. Disabled entirely (tests, local
    experimentation) via RATE_LIMIT_ENABLED=false.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    max_requests = limit if limit is not None else settings.RATE_LIMIT_RPM
    now = time.monotonic()
    cutoff = now - _WINDOW_SECONDS

    with _lock:
        if len(_windows) > _PRUNE_THRESHOLD:
            for stale_key in [k for k, w in _windows.items() if not w or w[-1] < cutoff]:
                del _windows[stale_key]

        window = _windows.setdefault(key, deque())
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= max_requests:
            retry_after = max(1, int(window[0] - cutoff) + 1)
            increment("rate_limit.rejections")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Slow down.",
                headers={"Retry-After": str(retry_after)},
            )

        window.append(now)


def reset_for_tests() -> None:
    with _lock:
        _windows.clear()
