"""
In-process operational counters.

Phase 6's observability floor, not its ceiling: before this, a total CrewAI
outage looked like a healthy service returning slightly worse answers, because
`direct_fallback` was a log line and nothing counted it.

Deliberately NOT a Prometheus client. This process has no Redis and no metrics
infrastructure (CLAUDE.md: the cache/rate-limit pools are no-op mocks); adding
a /metrics scrape format for infrastructure that does not exist would be
decoration. Flat thread-safe counters and a JSON snapshot at
`/health/metrics` make the routing mix, cache hit rate, and fallback rate
OBSERVABLE today, and the names are chosen so a later Prometheus exporter is a
transport change, not a renaming.

Counters are per-process and reset on restart. That is a documented property,
not a bug — trend analysis belongs to the persisted retrieval traces.
"""

from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_counters: dict[str, int] = {}
_started_at = time.time()


def increment(name: str, by: int = 1) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0) + by


def snapshot() -> dict:
    with _lock:
        counters = dict(sorted(_counters.items()))
    return {
        "uptime_seconds": int(time.time() - _started_at),
        "counters": counters,
        "scope": "per-process; resets on restart",
    }


def reset_for_tests() -> None:
    with _lock:
        _counters.clear()
