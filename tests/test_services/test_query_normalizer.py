"""
Unit tests for semantic-cache key normalization.

Fully deterministic — no LLM, no database, no network. These validate the only
thing the module actually produces: the normalized cache key. The predecessor
module also exposed `requires_isolation`, `detected_pii_types`, and
`original_query`; nothing in the application ever read them, so they were
removed (docs/ADVANCED_RAG_PLAN.md §2.3). The PII regexes still run — they are
asserted here through the masked key, which is their only observable effect.
"""

import json
from pathlib import Path

import pytest

from app.services.query_normalizer import normalize_for_cache_key

# ── Fixture-driven masking tests ────────────────────────────

FIXTURE_PATH = Path("tests/test_eval/fixtures/sanitizer_cases.json")
FIXTURE_CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case",
    [c for c in FIXTURE_CASES if not c.get("skip")],
    ids=[c["input"][:50] for c in FIXTURE_CASES if not c.get("skip")],
)
def test_detected_pii_is_masked_out_of_the_key(case: dict):
    """Every PII type a case declares must appear as a mask tag; a PII-free
    input must produce no mask tag at all."""
    result = normalize_for_cache_key(case["input"])

    for pii_type in case["expected_pii"]:
        assert f"[{pii_type.lower()}]" in result, (
            f"Expected {pii_type} to be masked in '{case['input']}' — got '{result}'"
        )

    if not case["expected_pii"]:
        assert "[" not in result, f"Unexpected mask tag in '{result}' for a PII-free input"


@pytest.mark.parametrize(
    "case",
    [c for c in FIXTURE_CASES if "expected_normalized" in c],
    ids=[c["input"][:50] for c in FIXTURE_CASES if "expected_normalized" in c],
)
def test_normalization_exact(case: dict):
    assert normalize_for_cache_key(case["input"]) == case["expected_normalized"]


@pytest.mark.parametrize(
    "case",
    [c for c in FIXTURE_CASES if "expected_normalized_contains" in c],
    ids=[c["input"][:50] for c in FIXTURE_CASES if "expected_normalized_contains" in c],
)
def test_normalization_contains_tag(case: dict):
    assert case["expected_normalized_contains"] in normalize_for_cache_key(case["input"])


# ── Explicit edge case tests ────────────────────────────────

def test_raw_pii_never_reaches_the_key():
    """The cache key is persisted on the :SemanticCache node, so raw
    identifiers must not survive into it — that is the whole point of masking."""
    result = normalize_for_cache_key("Email me at test@example.com about SSN 123-45-6789")
    assert "test@example.com" not in result
    assert "123-45-6789" not in result
    assert "[email]" in result
    assert "[ssn]" in result


def test_multiple_pii_types_masked():
    result = normalize_for_cache_key("Email john@test.com from IP 10.0.0.1")
    assert "[email]" in result
    assert "[ipv4]" in result


def test_whitespace_normalization():
    assert normalize_for_cache_key("  what   is    machine   learning  ") == "what is machine learning"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hey, what is AI?", "what is ai?"),
        ("hi, tell me about GPT", "about gpt"),
        ("please explain RAG", "rag"),
        ("can you tell me about vectors", "about vectors"),
        ("i want to know about embeddings", "about embeddings"),
    ],
)
def test_filler_prefix_stripping(raw: str, expected: str):
    assert normalize_for_cache_key(raw) == expected


def test_idempotent_normalization():
    """Re-normalizing a key must be a no-op, or cache lookups would drift."""
    first = normalize_for_cache_key("Hey, what is AI?")
    assert normalize_for_cache_key(first) == first


def test_masking_collides_distinct_queries():
    """Documents a known, accepted limitation rather than asserting it is good.

    Two different emails produce the same key, so these queries share a cache
    entry and can serve each other's answer. Entries are user-scoped, so this is
    not a cross-tenant leak. Phase 2 splits retrieval text from the cache key
    and is expected to make this assertion fail — update it there deliberately.
    """
    a = normalize_for_cache_key("is alice@corp.com blocked")
    b = normalize_for_cache_key("is bob@other.org blocked")
    assert a == b == "is [email] blocked"
