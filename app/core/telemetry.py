"""
OpenTelemetry + LangSmith tracing for CrewAI, OpenAI, and Google GenAI.

This module sets up the OpenTelemetry TracerProvider, attaches the
LangSmith OtelSpanProcessor (which auto-exports spans to your LangSmith
project), and instruments the AI libraries so every LLM call, tool
invocation, and crew kickoff is traced end-to-end.

Usage:
    Call ``init_telemetry()`` once at application startup (e.g. in the
    FastAPI lifespan).  Call ``shutdown_telemetry()`` on shutdown.

Required env vars (in .env):
    LANGSMITH_API_KEY   — your LangSmith API key
    LANGSMITH_PROJECT   — project name (defaults to "contextflow_dev")
    LANGCHAIN_TRACING_V2=true  — enables LangSmith tracing
"""

from __future__ import annotations

import logging
import os

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from app.core.config import settings

logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None


def init_telemetry() -> None:
    """Initialise OpenTelemetry + LangSmith tracing.

    Safe to call multiple times — subsequent calls are no-ops.
    Silently skips setup if LANGSMITH_API_KEY is not configured.
    """
    global _tracer_provider

    if _tracer_provider is not None:
        return  # already initialised

    api_key = settings.LANGSMITH_API_KEY.get_secret_value()
    if not api_key:
        # Fail LOUD outside local (Phase 6): a production deploy with a
        # missing key previously ran fully untraced behind one info line.
        if settings.ENVIRONMENT.value == "local":
            logger.info("LANGSMITH_API_KEY not set — skipping telemetry setup")
        else:
            logger.error(
                "LANGSMITH_API_KEY not set in %r — this deployment is running "
                "UNTRACED. Every LLM call, crew run, and failure is invisible.",
                settings.ENVIRONMENT.value,
            )
        return

    # LangSmith reads these env vars directly
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)

    # ── TracerProvider ──────────────────────────────────────
    from langsmith.integrations.otel import OtelSpanProcessor

    _tracer_provider = TracerProvider()
    _tracer_provider.add_span_processor(OtelSpanProcessor())
    trace.set_tracer_provider(_tracer_provider)

    # ── Instrument AI libraries ─────────────────────────────
    _instrument_crewai()
    _instrument_openai()
    _instrument_google_genai()

    logger.info(
        "✅ Telemetry initialised — LangSmith project: %s",
        settings.LANGSMITH_PROJECT,
    )


def _instrument_crewai() -> None:
    try:
        from openinference.instrumentation.crewai import CrewAIInstrumentor

        CrewAIInstrumentor().instrument()
        logger.debug("CrewAI instrumented")
    except Exception as exc:
        logger.warning("Could not instrument CrewAI: %s", exc)


def _instrument_openai() -> None:
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
        logger.debug("OpenAI instrumented")
    except Exception as exc:
        logger.warning("Could not instrument OpenAI: %s", exc)


def _instrument_google_genai() -> None:
    if settings.LLM_PROVIDER != "google":
        return
    try:
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

        GoogleGenAIInstrumentor().instrument()
        logger.debug("Google GenAI instrumented")
    except Exception as exc:
        logger.warning("Could not instrument Google GenAI: %s", exc)


def shutdown_telemetry() -> None:
    """Flush and shut down the tracer provider."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
        logger.info("Telemetry shut down")
