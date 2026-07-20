"""
Cache-key normalization for the Neo4j semantic cache.

This module is NOT a privacy control, despite what its predecessor's docstring
claimed. The raw user query still reaches the LLM provider, the CrewAI kickoff,
and Mem0's long-term store. Masking here changes only the cache key. See
docs/ADVANCED_RAG_PLAN.md §2.3 — a real PII policy is a separate decision.

PII patterns are masked for one narrow reason: `normalized_query` is persisted
verbatim as a property on the `:SemanticCache` node, so masking keeps raw
identifiers out of Neo4j at rest. It buys nothing upstream of that.

KNOWN LIMITATION — masking is lossy, so two distinct queries can collapse to the
same key ("is a@x.com blocked" and "is b@y.com blocked" both become
"is [email] blocked") and will serve each other's cached answer. Cache entries
are user-scoped, so this is not a cross-tenant leak, but it is still wrong.
Phase 2 of the plan splits this into a lossless retrieval normalizer and a
separate cache key; the collision is fixed there, not here.
"""

import re

# Regex mappings used to keep raw identifiers out of the persisted cache key.
PII_REGEX_PATTERNS = {
    "EMAIL": r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    "PHONE": r"\b(?:\+?1[-.\s]?)?\(?[2-9][0-8][0-9]\)?[-.\s]?[2-9][0-9]{2}[-.\s]?[0-9]{4}\b",
    "SSN": r"\b(?!000|666)[0-8][0-9]{2}-(?!00)[0-9]{2}-(?!0000)[0-9]{4}\b",
    "CREDIT_CARD": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})\b",
    "IPV4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
}

# Conversational openers stripped so "hey, what is X" and "what is X" share a
# cache entry. These are safe on a cache key but NOT on retrieval text — some of
# them ("please explain") carry intent. Phase 2 keeps them out of the retrieval
# normalizer for that reason.
_FILLER_PREFIXES = [
    r"^hey,?\s*",
    r"^hi,?\s*",
    r"^hello,?\s*",
    r"^can you tell me\s+",
    r"^tell me\s+",
    r"^i want to know\s+",
    r"^please explain\s+",
    r"^please\s+",
]


def normalize_for_retrieval(query: str) -> str:
    """Light normalization for the text that actually gets searched.

    Deliberately does far less than `normalize_for_cache_key`:

    - NO filler stripping. "please explain X" and "X" are different requests,
      and "can you tell me about vectors" -> "about vectors" mangles the query
      into a fragment. Modern encoders do not need those crutches, and the
      stripping regexes change meaning.
    - NO lowercasing. Case carries signal for proper nouns, acronyms, and
      identifiers, all of which are exactly what factual lookups turn on.
    - NO PII masking. Masking is lossy; replacing an identifier with "[EMAIL]"
      destroys the one term that would have matched.

    A cache key wants aggressive collapsing so near-identical queries share an
    entry. Retrieval wants the opposite. Using one function for both was a
    category error — this split is the fix.
    """
    return re.sub(r"\s+", " ", query.strip())


def normalize_for_cache_key(query: str) -> str:
    """Reduce a raw query to a stable semantic-cache key.

    Collapses whitespace, masks PII patterns, lowercases, and strips
    conversational filler prefixes. Lossy by design — see the module docstring.
    """
    normalized = re.sub(r"\s+", " ", query.strip())

    # Mask before lowercasing so the patterns match their original casing.
    for pii_type, pattern in PII_REGEX_PATTERNS.items():
        normalized = re.sub(pattern, f"[{pii_type}]", normalized)

    normalized = normalized.lower()

    for filler in _FILLER_PREFIXES:
        normalized = re.sub(filler, "", normalized)

    return normalized.strip()
