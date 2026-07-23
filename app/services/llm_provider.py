"""
Multi-provider LLM configuration for CrewAI.

Supported providers (set via LLM_PROVIDER env var):
  - "openai"     → Direct OpenAI API (or any OpenAI-compatible endpoint)
  - "google"     → Google Gemini via native CrewAI support
  - "openrouter" → OpenRouter via LiteLLM prefix (openrouter/model-name)

Usage:
    from app.services.llm_provider import build_crewai_llm
    llm = build_crewai_llm()
"""

from __future__ import annotations

from crewai import LLM

from app.core.config import settings


def get_openai_compatible_kwargs() -> dict:
    """Build kwargs for any OpenAI-compatible client (embeddings, direct calls)."""
    kwargs: dict = {}
    api_key = settings.OPENAI_API_KEY.get_secret_value()
    if api_key:
        kwargs["api_key"] = api_key
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    return kwargs


def build_crewai_llm() -> LLM:
    """Build a CrewAI-compatible LLM based on the configured provider."""
    provider = settings.LLM_PROVIDER
    model = settings.LLM_MODEL

    if provider == "openai":
        # Native OpenAI — pass model as-is (e.g. "gpt-4.1-mini")
        return LLM(
            model=model,
            temperature=0,
            **get_openai_compatible_kwargs(),
        )

    if provider == "google":
        # Native Google Gemini — CrewAI recognizes "gemini/" prefix
        prefixed = model if model.startswith("gemini/") else f"gemini/{model}"
        return LLM(
            model=prefixed,
            temperature=0,
            api_key=settings.GOOGLE_API_KEY.get_secret_value(),
        )

    if provider == "openrouter":
        # OpenRouter via LiteLLM — needs "openrouter/" prefix
        prefixed = model if model.startswith("openrouter/") else f"openrouter/{model}"
        return LLM(
            model=prefixed,
            temperature=0,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL or "https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'. Supported: 'openai', 'google', 'openrouter'")


def build_crewai_embedder() -> dict:
    """Build the embedder config dict for CrewAI's internal memory system.

    CrewAI's ``Crew(memory=True)`` defaults to OpenAI embeddings.
    This function returns an ``embedder`` dict that matches the
    configured EMBEDDING_PROVIDER so CrewAI's memory works with
    whichever embedding backend is active.

    Usage:
        Crew(memory=True, embedder=build_crewai_embedder())
    """
    provider = settings.EMBEDDING_PROVIDER

    if provider == "openai":
        config: dict = {
            "model": settings.EMBEDDING_MODEL,
        }
        api_key = settings.OPENAI_API_KEY.get_secret_value()
        if api_key:
            config["api_key"] = api_key
        if settings.OPENAI_BASE_URL:
            config["api_base"] = settings.OPENAI_BASE_URL
        return {"provider": "openai", "config": config}

    if provider == "huggingface":
        return {
            "provider": "huggingface",
            "config": {"model": settings.EMBEDDING_MODEL},
        }
    
    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER for CrewAI memory: '{provider}'. "
        f"Supported: 'openai', 'huggingface'"
    )


# ── Direct LLM Client (bypasses CrewAI) ─────────────────────


def get_async_llm_client():
    """Return an AsyncOpenAI client configured for the active LLM_PROVIDER.

    All three providers (openai, google, openrouter) support the
    OpenAI-compatible chat completions API, so we use the same
    ``openai.AsyncOpenAI`` client for all of them.

    - **openai**: Uses OPENAI_API_KEY + optional OPENAI_BASE_URL.
    - **google**: Uses the Gemini OpenAI-compat endpoint at
      ``generativelanguage.googleapis.com/v1beta/openai/``.
    - **openrouter**: Uses OPENAI_API_KEY + ``openrouter.ai/api/v1``.
    """
    from openai import AsyncOpenAI

    provider = settings.LLM_PROVIDER

    if provider == "openai":
        return AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL or None,
        )

    if provider == "google":
        return AsyncOpenAI(
            api_key=settings.GOOGLE_API_KEY.get_secret_value(),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    if provider == "openrouter":
        return AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL or "https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: '{provider}'")


def _get_model_name() -> str:
    """Return the model name suitable for the OpenAI-compatible client."""
    provider = settings.LLM_PROVIDER
    model = settings.LLM_MODEL

    if provider == "google":
        # Gemini OpenAI-compat expects bare model name (no "gemini/" prefix)
        return model.removeprefix("gemini/")

    # openai / openrouter: model name as-is
    return model


async def classify_intent(query: str) -> bool:
    """Fast intent classification — returns True if the query needs RAG.

    Uses a two-layer strategy:
    1. **Keyword heuristic** — catches obvious greetings/small-talk in 0ms.
    2. **LLM classification** — for ambiguous queries, uses a lightweight
       LLM call (~200ms) with text output (no JSON mode for provider compat).

    Returns False for simple chat, True for knowledge queries.
    """
    import logging

    logger = logging.getLogger(__name__)

    # ── Layer 1: Fast keyword heuristic (0ms) ──────────────
    query_lower = query.lower().strip()
    chat_phrase = query_lower.strip(" \t\r\n.,!?")

    # Obvious greetings / small talk — never need RAG
    SIMPLE_PATTERNS = {
        "hi", "hello", "hey", "howdy", "yo", "sup",
        "thanks", "thank you", "thx", "ty",
        "bye", "goodbye", "see ya", "later",
        "good morning", "good evening", "good night",
        "how are you", "what's up", "whats up",
        "ok", "okay", "sure", "yes", "no", "yep", "nope",
        "lol", "haha", "nice", "cool", "great", "awesome",
    }

    if chat_phrase in SIMPLE_PATTERNS:
        logger.info("Intent [heuristic]: simple chat — '%s'", query[:60])
        return False

    # ── Layer 2: LLM classification (ambiguous queries) ────
    client = get_async_llm_client()
    # One-word output; the cheap classifier model if configured (Phase 6).
    model = settings.CLASSIFIER_MODEL or _get_model_name()

    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=10,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intent classifier. "
                        "Reply with EXACTLY one word: RAG or CHAT.\n"
                        "Say RAG if the user needs factual info, document lookup, "
                        "or knowledge retrieval.\n"
                        "Say CHAT if it's a greeting, small talk, opinion, "
                        "thanks, or casual conversation."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        answer = response.choices[0].message.content.strip().upper()
        if answer == "RAG":
            needs_rag = True
        elif answer == "CHAT":
            needs_rag = False
        else:
            logger.warning("Intent classifier returned unexpected output %r, defaulting to RAG", answer)
            needs_rag = True
        logger.info(
            "Intent [LLM]: %s — '%s' (raw: %s)",
            "RAG" if needs_rag else "CHAT",
            query[:60],
            answer,
        )
        return needs_rag
    except Exception as exc:
        logger.warning("Intent classification failed (%s), defaulting to RAG", exc)
        return True


async def stream_direct_chat(
    query: str,
    *,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
):
    """Stream a direct LLM response, bypassing CrewAI entirely.

    Yields content chunks as they arrive. Use for simple chat queries
    that don't need RAG tool invocation.

    Args:
        query: The user's message.
        system_prompt: Optional system prompt override.
        history: Optional list of prior messages [{"role": ..., "content": ...}].

    Yields:
        str: Content delta chunks from the streaming response.
    """
    client = get_async_llm_client()
    model = _get_model_name()

    messages: list[dict] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    else:
        messages.append({
            "role": "system",
            "content": (
                "You are a helpful, friendly AI assistant. "
                "Keep answers concise and conversational."
            ),
        })

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": query})

    stream = await client.chat.completions.create(
        model=model,
        temperature=0.7,
        stream=True,
        # Backstop against runaway generations, not a style constraint —
        # prompts already ask for short conversational answers (Phase 6).
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=messages,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def direct_chat(
    query: str,
    *,
    system_prompt: str | None = None,
    history: list[dict] | None = None,
) -> str:
    """Non-streaming direct LLM call. Returns the full response string."""
    chunks: list[str] = []
    async for chunk in stream_direct_chat(
        query, system_prompt=system_prompt, history=history
    ):
        chunks.append(chunk)
    return "".join(chunks)
