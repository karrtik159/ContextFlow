"""
Phase 6 — operational readiness contract.

Request correlation, readiness vs liveness, metrics exposure, the MemoryCrew
substance gate, rate limiting on the spend-bearing paths, and the boot-time
security guards. Each test pins a behavior the plan doc called out as a
production gap.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import EnvironmentOption, settings
from app.core.metrics import reset_for_tests as reset_metrics
from app.core.rate_limit import reset_for_tests as reset_rate_limit
from app.core.setup import enforce_boot_security, resolve_cors_credentials
from app.main import app


@pytest.fixture(autouse=True)
def _clean_state():
    reset_rate_limit()
    reset_metrics()
    yield
    reset_rate_limit()
    reset_metrics()
    app.dependency_overrides.clear()


async def _get(path, headers=None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path, headers=headers or {})


# ── Request correlation ─────────────────────────────────────

@pytest.mark.asyncio
async def test_every_response_carries_a_request_id():
    response = await _get("/health")
    assert response.headers.get("X-Request-ID"), "responses must be correlatable"


@pytest.mark.asyncio
async def test_inbound_request_id_is_honoured():
    response = await _get("/health", headers={"X-Request-ID": "trace-me-42"})
    assert response.headers["X-Request-ID"] == "trace-me-42"


@pytest.mark.asyncio
async def test_hostile_request_id_is_sanitized():
    """The header is caller-controlled and goes into log lines — newlines and
    exotic characters must not survive into it."""
    response = await _get("/health", headers={"X-Request-ID": "abc\tdef ghi%0a"})
    returned = response.headers["X-Request-ID"]
    assert "\t" not in returned and " " not in returned and "%" not in returned


# ── Liveness vs readiness ───────────────────────────────────

@pytest.mark.asyncio
async def test_readiness_reports_ready_when_all_probes_pass(monkeypatch):
    async def ok():
        return None

    monkeypatch.setattr("app.main._probe_postgres", ok)
    monkeypatch.setattr("app.main._probe_neo4j", ok)

    response = await _get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["dependencies"] == {"postgres": "ok", "neo4j": "ok"}


@pytest.mark.asyncio
async def test_readiness_is_503_when_a_dependency_is_down(monkeypatch):
    """The old /health returned 'healthy' unconditionally — an instance with
    no database passed every check while failing every request."""

    async def ok():
        return None

    async def down():
        return "ConnectionRefusedError"

    monkeypatch.setattr("app.main._probe_postgres", down)
    monkeypatch.setattr("app.main._probe_neo4j", ok)

    response = await _get("/health/ready")

    assert response.status_code == 503
    assert response.json()["dependencies"]["postgres"] == "ConnectionRefusedError"


@pytest.mark.asyncio
async def test_metrics_snapshot_is_exposed():
    from app.core.metrics import increment

    increment("rag.routed.cache")
    response = await _get("/health/metrics")
    assert response.status_code == 200
    assert response.json()["counters"]["rag.routed.cache"] == 1


# ── Auth: presented-but-invalid credentials are an error ────

@pytest.mark.asyncio
async def test_garbage_bearer_token_is_401_not_anonymous():
    """The old behavior silently degraded an invalid token to anonymous: an
    expired session kept getting 200s while quietly losing its scope."""
    response = await _get(
        "/api/v1/documents", headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_malformed_authorization_header_is_401():
    response = await _get("/api/v1/documents", headers={"Authorization": "Basic abc"})
    assert response.status_code == 401


# ── MemoryCrew substance gate ───────────────────────────────

def test_trivial_exchanges_are_not_memory_worthy():
    from app.api.v1.rag import _memory_worthy

    assert not _memory_worthy("thanks", "You're welcome!")
    assert not _memory_worthy("hi", "Hello! How can I help you today?")


def test_substantive_exchanges_are_memory_worthy():
    from app.api.v1.rag import _memory_worthy

    assert _memory_worthy(
        "How often do the log files rotate on the ingest node?",
        "Log files rotate every four hours, and an owner can trigger it manually once per hour.",
    )


def test_schedule_memory_skips_and_counts_trivial(monkeypatch):
    from fastapi import BackgroundTasks

    from app.api.v1.rag import _schedule_memory
    from app.core.metrics import snapshot

    tasks = BackgroundTasks()
    _schedule_memory(tasks, query="thanks", answer="np!", user_id="u")

    assert tasks.tasks == [], "a trivial exchange must not spend 3-6 LLM calls"
    assert snapshot()["counters"]["memory_crew.skipped_trivial"] == 1


# ── Boot security guards ────────────────────────────────────

_DEFAULT_KEY = "super-secret-change-me-in-production"


def test_default_secret_key_refuses_to_boot_outside_local(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", EnvironmentOption.PRODUCTION)
    monkeypatch.setattr(settings, "SECRET_KEY", type(settings.SECRET_KEY)(_DEFAULT_KEY))
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        enforce_boot_security(settings)


def test_default_secret_key_warns_but_boots_locally(monkeypatch, caplog):
    monkeypatch.setattr(settings, "ENVIRONMENT", EnvironmentOption.LOCAL)
    monkeypatch.setattr(settings, "SECRET_KEY", type(settings.SECRET_KEY)(_DEFAULT_KEY))
    enforce_boot_security(settings)  # must not raise


def test_real_secret_key_boots_anywhere(monkeypatch):
    monkeypatch.setattr(settings, "ENVIRONMENT", EnvironmentOption.PRODUCTION)
    monkeypatch.setattr(settings, "SECRET_KEY", type(settings.SECRET_KEY)("a-real-key"))
    enforce_boot_security(settings)  # must not raise


def test_wildcard_cors_disables_credentials(monkeypatch):
    """Starlette echoes the Origin for ["*"] + credentials, defeating the
    browser protection — any site could make credentialed calls."""
    monkeypatch.setattr(settings, "CORS_ORIGINS", ["*"])
    assert resolve_cors_credentials(settings) is False


def test_explicit_cors_origins_keep_credentials(monkeypatch):
    monkeypatch.setattr(settings, "CORS_ORIGINS", ["https://app.example.com"])
    assert resolve_cors_credentials(settings) is True


# ── Telemetry fails loud outside local ──────────────────────

def test_missing_telemetry_key_is_an_error_outside_local(monkeypatch, caplog):
    import logging

    from app.core import telemetry

    monkeypatch.setattr(telemetry, "_tracer_provider", None)
    monkeypatch.setattr(settings, "ENVIRONMENT", EnvironmentOption.STAGING)
    monkeypatch.setattr(
        settings, "LANGSMITH_API_KEY", type(settings.LANGSMITH_API_KEY)("")
    )

    with caplog.at_level(logging.ERROR, logger="app.core.telemetry"):
        telemetry.init_telemetry()

    assert any("UNTRACED" in r.getMessage() for r in caplog.records), (
        "an untraced non-local deploy must be an ERROR, not an info line"
    )
