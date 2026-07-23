import pytest

from app.services.semantic_cache import (
    cache_version,
    get_cached_response,
    invalidate_user_cache,
    populate_semantic_cache,
)


class _FakeResult:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record


class _FakeSession:
    def __init__(self, records):
        self.records = list(records)
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def run(self, query, **params):
        self.calls.append({"query": query, "params": params})
        record = self.records.pop(0) if self.records else None
        return _FakeResult(record)


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


@pytest.mark.asyncio
async def test_get_cached_response_exact_match_is_user_scoped(monkeypatch):
    session = _FakeSession(records=[{"answer": "Exact scoped answer"}])

    async def fake_driver():
        return _FakeDriver(session)

    monkeypatch.setattr("app.services.semantic_cache.get_driver", fake_driver)

    answer = await get_cached_response(
        normalized_query="what did i decide?",
        embedding=[0.1, 0.2, 0.3],
        user_id="user-a",
    )

    assert answer == "Exact scoped answer"
    assert len(session.calls) == 1
    assert "user_id: $user_id" in session.calls[0]["query"]
    assert session.calls[0]["params"]["user_id"] == "user-a"


@pytest.mark.asyncio
async def test_get_cached_response_vector_match_filters_by_user(monkeypatch):
    session = _FakeSession(records=[None, {"answer": "Vector scoped answer"}])

    async def fake_driver():
        return _FakeDriver(session)

    monkeypatch.setattr("app.services.semantic_cache.get_driver", fake_driver)

    answer = await get_cached_response(
        normalized_query="what did i decide?",
        embedding=[0.1, 0.2, 0.3],
        user_id="user-b",
    )

    assert answer == "Vector scoped answer"
    assert len(session.calls) == 2
    vector_call = session.calls[1]
    assert "c.user_id = $user_id" in vector_call["query"]
    assert vector_call["params"]["user_id"] == "user-b"
    assert vector_call["params"]["candidate_count"] == 50


@pytest.mark.asyncio
async def test_populate_semantic_cache_upserts_per_user_query(monkeypatch):
    session = _FakeSession(records=[None])

    async def fake_driver():
        return _FakeDriver(session)

    monkeypatch.setattr("app.services.semantic_cache.get_driver", fake_driver)

    await populate_semantic_cache(
        normalized_query="what did i decide?",
        embedding=[0.1, 0.2, 0.3],
        answer="Scoped answer",
        user_id="user-c",
        session_id="session-1",
    )

    assert len(session.calls) == 1
    call = session.calls[0]
    assert "MERGE (c:SemanticCache" in call["query"]
    assert "CREATE (c:SemanticCache" not in call["query"]
    assert call["params"]["user_id"] == "user-c"
    assert call["params"]["session_id"] == "session-1"


# ── Phase 7: epoch / version / TTL validity ─────────────────


@pytest.mark.asyncio
async def test_lookups_filter_on_epoch_version_and_ttl(monkeypatch):
    """Both lookup shapes must carry all three validity predicates. The
    coalesce(-1) form is what makes a pre-Phase-7 node — which has no epoch
    property — fail against every real epoch instead of matching NULL-ishly."""
    session = _FakeSession(records=[None, None])

    async def fake_driver():
        return _FakeDriver(session)

    monkeypatch.setattr("app.services.semantic_cache.get_driver", fake_driver)

    answer = await get_cached_response(
        normalized_query="q",
        embedding=[0.1],
        user_id="user-d",
        corpus_epoch=4,
    )

    assert answer is None
    assert len(session.calls) == 2
    for call in session.calls:
        assert "coalesce(c.corpus_epoch, -1) = $corpus_epoch" in call["query"]
        assert "coalesce(c.cache_version, '') = $cache_version" in call["query"]
        assert "coalesce(c.timestamp, 0) >= $min_timestamp" in call["query"]
        assert call["params"]["corpus_epoch"] == 4
        assert call["params"]["cache_version"] == cache_version()
        assert call["params"]["min_timestamp"] > 0


@pytest.mark.asyncio
async def test_populate_stamps_epoch_and_version(monkeypatch):
    """An entry is stamped with the epoch the answer was RETRIEVED at, so a
    corpus mutation between retrieval and this background task cannot label a
    stale answer current."""
    session = _FakeSession(records=[None])

    async def fake_driver():
        return _FakeDriver(session)

    monkeypatch.setattr("app.services.semantic_cache.get_driver", fake_driver)

    await populate_semantic_cache(
        normalized_query="q",
        embedding=[0.1],
        answer="a",
        user_id="user-e",
        corpus_epoch=9,
    )

    call = session.calls[0]
    assert "c.corpus_epoch = $corpus_epoch" in call["query"]
    assert "c.cache_version = $cache_version" in call["query"]
    assert call["params"]["corpus_epoch"] == 9
    assert call["params"]["cache_version"] == cache_version()


@pytest.mark.asyncio
async def test_invalidate_user_cache_is_user_scoped(monkeypatch):
    session = _FakeSession(records=[None])

    async def fake_driver():
        return _FakeDriver(session)

    monkeypatch.setattr("app.services.semantic_cache.get_driver", fake_driver)

    await invalidate_user_cache("user-f")

    call = session.calls[0]
    assert "DETACH DELETE c" in call["query"]
    assert "user_id: $user_id" in call["query"]
    assert call["params"]["user_id"] == "user-f"
