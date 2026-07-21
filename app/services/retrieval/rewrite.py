"""
Query rewriting — turning one user question into the queries each arm wants.

The dense and sparse arms want different text, and the previous design gave
them the same string. An embedding model wants a natural, complete question;
`websearch_to_tsquery` wants terms, and it wants an identifier like
`20260714-af` kept intact as a phrase rather than stemmed into fragments.

Everything in `deterministic_rewrite` is free — no model, no network, no LLM
call. That matters: Phase 2's whole point was reducing a knowledge query to a
single LLM call, and a rewriting step that costs a round-trip would hand a
third of that budget back for an unmeasured gain. HyDE and multi-query DO cost
a call each, so they are behind flags that default to off and must earn
themselves against the golden set before being switched on.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)

# An "identifier" is a token dense retrieval is systematically bad at: it has
# no useful embedding neighbourhood because it is a name, not a concept.
#   20260714-af   release tag        E1042  error code
#   v2.14.0       version            SAML   acronym
_IDENTIFIER_PATTERNS = (
    # Contains a digit AND a non-digit: version strings, tags, error codes.
    re.compile(r"\b(?=[\w.-]*\d)(?=[\w.-]*[A-Za-z])[A-Za-z0-9][\w.-]{2,}\b"),
    # Two or more capitals in a row: SAML, HTTP, API, SSO.
    re.compile(r"\b[A-Z]{2,}\b"),
    # snake_case / dotted paths: hnsw.ef_search, user_id
    re.compile(r"\b[a-z][a-z0-9]*(?:[._][a-z0-9]+)+\b"),
)

# Explicitly quoted spans are the user telling us they mean it literally.
_QUOTED = re.compile(r"[\"'`]([^\"'`]{2,64})[\"'`]")


@dataclass
class RewrittenQuery:
    """What each arm should actually search for.

    `dense_query` deliberately defaults to the input untouched. Phase 2 already
    established that aggressive normalization belongs on the CACHE key, not the
    retrieval path: stripping "please explain" changes meaning, and modern
    encoders do not need the help. Rewriting the dense query is a change that
    must be measured, not assumed.
    """

    original: str
    dense_query: str
    sparse_query: str
    identifiers: list[str] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    hyde_document: str | None = None
    methods: list[str] = field(default_factory=list)

    @property
    def dense_queries(self) -> list[str]:
        """Every text the dense arm should embed — primary first."""
        texts = [self.dense_query]
        if self.hyde_document:
            texts.append(self.hyde_document)
        texts.extend(self.variants)
        return texts


def extract_identifiers(query: str, *, limit: int = 6) -> list[str]:
    """Pull out tokens that must survive into the lexical query intact."""
    found: list[str] = []
    seen: set[str] = set()

    for match in _QUOTED.finditer(query):
        value = match.group(1).strip()
        if value and value.casefold() not in seen:
            seen.add(value.casefold())
            found.append(value)

    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(query):
            value = match.group(0)
            if value.casefold() in seen:
                continue
            seen.add(value.casefold())
            found.append(value)

    return found[:limit]


def build_sparse_query(query: str, identifiers: list[str]) -> str:
    """Assemble the string handed to the sparse arm.

    This deliberately returns the query text unchanged.

    An earlier version appended each identifier back as a double-quoted phrase,
    on the theory that `websearch_to_tsquery` would treat it as an exact match
    and stop `20260714-af` being split. That reasoning does not survive contact
    with the actual consumer. `search_chunks_sparse` does not use
    `websearch_to_tsquery` — it lexes the whole string with `to_tsvector` and
    ORs the resulting lexemes. `to_tsvector` is a DOCUMENT parser: double
    quotes are punctuation to it, not phrase delimiters, and
    `tsvector_to_array` then de-duplicates, so the appended copy collapsed into
    the lexemes already there.

    Measured against PostgreSQL 16: the executed tsquery is byte-identical with
    and without the quoting. It was a stage that read as if it shaped the
    search and did not. `identifiers` is still extracted and recorded in the
    trace, where it is honest observability rather than a claimed effect.

    Making identifiers genuinely weightier needs a different mechanism —
    Postgres FTS ranking has no IDF — and belongs behind a measurement, not
    here. The reranker currently supplies that precision downstream.
    """
    return query.strip()


def deterministic_rewrite(query: str) -> RewrittenQuery:
    """The free path. No LLM, no network, no model load."""
    identifiers = extract_identifiers(query) if settings.QUERY_EXPANSION_ENABLED else []
    return RewrittenQuery(
        original=query,
        dense_query=query,
        sparse_query=build_sparse_query(query, identifiers),
        identifiers=identifiers,
        methods=["deterministic"] if settings.QUERY_EXPANSION_ENABLED else [],
    )


_HYDE_PROMPT = (
    "Write a short, factual passage (2-4 sentences) that would answer the "
    "question below, as if excerpted from internal product documentation. "
    "Invent plausible specifics rather than hedging — this text is used only "
    "as a retrieval probe and is never shown to a user or treated as an "
    "answer.\n\nQuestion: {query}\n\nPassage:"
)

_MULTI_QUERY_PROMPT = (
    "Rewrite the question below as {n} alternative search queries that a "
    "documentation search engine would match. Vary the vocabulary and phrasing; "
    "keep any identifiers, error codes, or product names exactly as written. "
    "Output one query per line, no numbering, no commentary.\n\n"
    "Question: {query}"
)


async def _complete(prompt: str, *, max_tokens: int) -> str:
    from app.services.llm_provider import _get_model_name, get_async_llm_client

    client = get_async_llm_client()
    response = await client.chat.completions.create(
        model=_get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return (response.choices[0].message.content or "").strip()


async def generate_hyde_document(query: str) -> str | None:
    """HyDE: embed a hypothetical ANSWER instead of the question.

    Questions and answers live in different regions of embedding space — "how
    do I roll back?" looks less like the rollback documentation than a
    fabricated rollback paragraph does. Costs one LLM call, which is why it is
    flag-gated.

    The generated text is a RETRIEVAL PROBE and never reaches the user. It is
    deliberately allowed to be wrong; the corpus, not this text, is what the
    answer gets grounded in.
    """
    if not settings.HYDE_ENABLED:
        return None
    try:
        text = await _complete(_HYDE_PROMPT.format(query=query), max_tokens=200)
    except Exception as exc:
        logger.warning("HyDE generation failed, continuing without it: %s", exc)
        return None
    return text or None


async def generate_query_variants(query: str) -> list[str]:
    """Multi-query expansion. One LLM call, flag-gated, degrades to []."""
    if not settings.MULTI_QUERY_ENABLED:
        return []
    n = max(1, settings.MULTI_QUERY_COUNT)
    try:
        text = await _complete(
            _MULTI_QUERY_PROMPT.format(n=n, query=query), max_tokens=200
        )
    except Exception as exc:
        logger.warning("Multi-query expansion failed, continuing without it: %s", exc)
        return []

    variants: list[str] = []
    seen = {query.casefold()}
    for line in text.splitlines():
        candidate = line.strip().lstrip("-*0123456789. ").strip()
        if len(candidate) < 3 or candidate.casefold() in seen:
            continue
        seen.add(candidate.casefold())
        variants.append(candidate)
    return variants[:n]


async def rewrite_query(query: str) -> RewrittenQuery:
    """Full rewrite: deterministic always, HyDE and multi-query if enabled."""
    rewritten = deterministic_rewrite(query)

    if settings.HYDE_ENABLED:
        hyde = await generate_hyde_document(query)
        if hyde:
            rewritten.hyde_document = hyde
            rewritten.methods.append("hyde")

    if settings.MULTI_QUERY_ENABLED:
        variants = await generate_query_variants(query)
        if variants:
            rewritten.variants = variants
            rewritten.methods.append("multi_query")

    return rewritten
