"""
Sliding-window rate limiter (Phase 6).

In-process per-tenant limiter on the spend-bearing paths. These tests pin the
window arithmetic, the flag, the Retry-After contract, and per-key isolation.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.config import settings
from app.core.rate_limit import check_rate_limit, reset_for_tests


@pytest.fixture(autouse=True)
def _clean():
    reset_for_tests()
    yield
    reset_for_tests()


def test_requests_under_the_limit_pass():
    for _ in range(5):
        check_rate_limit("user-a", limit=5)


def test_request_over_the_limit_is_429_with_retry_after():
    for _ in range(3):
        check_rate_limit("user-a", limit=3)

    with pytest.raises(HTTPException) as excinfo:
        check_rate_limit("user-a", limit=3)

    assert excinfo.value.status_code == 429
    assert int(excinfo.value.headers["Retry-After"]) >= 1


def test_keys_are_isolated():
    for _ in range(3):
        check_rate_limit("user-a", limit=3)
    # A different tenant is untouched by user-a's burst.
    check_rate_limit("user-b", limit=3)


def test_disabled_flag_bypasses_entirely(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    for _ in range(50):
        check_rate_limit("user-a", limit=1)


def test_default_limit_comes_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_RPM", 2)
    check_rate_limit("user-c")
    check_rate_limit("user-c")
    with pytest.raises(HTTPException):
        check_rate_limit("user-c")


def test_window_slides(monkeypatch):
    """Old entries age out: after the window passes, capacity returns."""
    import app.core.rate_limit as rl

    now = {"t": 1000.0}
    monkeypatch.setattr(rl.time, "monotonic", lambda: now["t"])

    check_rate_limit("user-d", limit=1)
    with pytest.raises(HTTPException):
        check_rate_limit("user-d", limit=1)

    now["t"] += 61.0
    check_rate_limit("user-d", limit=1)  # window slid; allowed again
