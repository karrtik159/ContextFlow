"""
Cross-encoder reranker — Phase 3.

Nothing here loads model weights. Two of these tests exist specifically to
guarantee that: `test_import_does_not_load_model` is the reranker's equivalent
of `test_graph_search_import_does_not_connect`, and it protects the same
property — unit tests must run on a machine with no model cache and no network.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from app.core.config import settings
from app.services import reranker as reranker_mod
from app.services.reranker import (
    RerankUnavailable,
    _sigmoid,
    rerank_async,
    rerank_sync,
    reset_reranker_for_tests,
)


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_reranker_for_tests()
    yield
    reset_reranker_for_tests()


def test_import_does_not_load_model():
    """Importing the module must not touch torch or sentence-transformers.

    A 400 MB import is an import-time landmine: it makes `import app` fail on a
    machine with no model cache, and it makes every unit test in the suite pay
    for a model that most of them never use. Same discipline as
    `test_graph_search_import_does_not_connect`.

    Run in a SUBPROCESS rather than via `importlib.reload`. Reloading rebinds
    every class in the module — including `RerankUnavailable` — so subsequent
    tests in this file would be catching a different exception object than the
    one the module raises, and would fail for a reason that has nothing to do
    with the code under test. A clean interpreter is also what the property
    actually claims.
    """
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import app.services.reranker as r; "
            "print(int('torch' in sys.modules), "
            "int('sentence_transformers' in sys.modules), "
            "int(r._cross_encoder is None))",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    torch_loaded, st_loaded, encoder_is_none = result.stdout.split()
    assert torch_loaded == "0", "importing reranker pulled in torch"
    assert st_loaded == "0", "importing reranker pulled in sentence_transformers"
    assert encoder_is_none == "1"


class TestSigmoid:
    def test_squashes_logits(self):
        assert _sigmoid(-4.0) < 0.05
        assert _sigmoid(8.0) > 0.99
        assert _sigmoid(0.0) == 0.5

    def test_output_is_always_in_unit_interval(self):
        for raw in (-1e6, -300.0, -5.5, 2.5, 300.0, 1e6):
            assert 0.0 <= _sigmoid(raw) <= 1.0

    def test_extreme_logits_do_not_overflow(self):
        """math.exp(1e6) raises OverflowError — a crash in the rerank stage."""
        assert _sigmoid(-1e9) == 0.0
        assert _sigmoid(1e9) == 1.0

    def test_is_monotonic(self):
        raws = [-100.0, -5.0, -1.0, -0.4, 0.0, 0.4, 0.9, 1.0, 1.5, 5.0, 100.0]
        scores = [_sigmoid(r) for r in raws]
        assert scores == sorted(scores)


class TestNormalizationIsDecidedPerModel:
    """Deciding the convention per SCORE rather than per MODEL is not
    monotonic. 'Pass through if already in [0, 1], else sigmoid' maps a logit
    of 0.9 to 0.9 but a LARGER logit of 1.5 to sigmoid(1.5) = 0.82, inverting
    the ranking of exactly the two candidates a reranker exists to separate.
    """

    def test_sigmoid_model_output_passes_through(self, monkeypatch):
        import torch.nn as nn

        class FakeEncoder:
            activation_fn = nn.Sigmoid()

            def predict(self, pairs):
                return [0.9, 0.3]

        monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: FakeEncoder())
        assert rerank_sync("q", ["a", "b"]) == [0.9, 0.3]

    def test_logit_model_output_is_squashed(self, monkeypatch):
        class FakeEncoder:
            activation_fn = None

            def predict(self, pairs):
                return [0.9, 1.5]

        monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: FakeEncoder())
        scores = rerank_sync("q", ["a", "b"])
        assert scores[1] > scores[0], "a larger logit must produce a larger score"

    def test_ordering_survives_the_unit_boundary(self, monkeypatch):
        """The regression this class exists for: raw values straddling 1.0."""

        class FakeEncoder:
            activation_fn = None

            def predict(self, pairs):
                return [0.5, 0.95, 1.05, 2.0, 4.0]

        monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: FakeEncoder())
        scores = rerank_sync("q", ["a", "b", "c", "d", "e"])
        assert scores == sorted(scores)

    def test_unrecognised_model_is_assumed_to_emit_logits(self, monkeypatch):
        class FakeEncoder:
            def predict(self, pairs):
                return [0.0]

        monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: FakeEncoder())
        assert rerank_sync("q", ["a"]) == [0.5]  # sigmoid(0) == 0.5, not 0.0

    def test_legacy_attribute_name_is_honoured(self, monkeypatch):
        """sentence-transformers renamed this across major versions."""
        import torch.nn as nn

        class FakeEncoder:
            default_activation_function = nn.Sigmoid()

            def predict(self, pairs):
                return [0.9]

        monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: FakeEncoder())
        assert rerank_sync("q", ["a"]) == [0.9]


class TestEmptyInput:
    def test_sync_returns_empty_without_loading(self):
        assert rerank_sync("q", []) == []
        assert reranker_mod._cross_encoder is None

    def test_async_returns_empty_without_loading(self):
        assert asyncio.run(rerank_async("q", [])) == []
        assert reranker_mod._cross_encoder is None


class _FakeSTModule:
    """Stand-in for `sentence_transformers` whose CrossEncoder is controllable."""

    def __init__(self, ctor):
        self.CrossEncoder = ctor


class TestLoadFailure:
    def test_load_failure_raises_rerank_unavailable(self, monkeypatch):
        monkeypatch.setattr(settings, "RERANK_MODEL", "definitely/not-a-real-model-xyz")

        def boom(*args, **kwargs):
            raise OSError("no such model")

        monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeSTModule(boom))
        with pytest.raises(RerankUnavailable):
            rerank_sync("q", ["doc"])

    def test_failure_is_latched_and_not_retried(self, monkeypatch):
        """A model that failed to load will not spontaneously start loading.

        Without the latch, every query in a process with a broken model cache
        re-attempts the load and re-pays the timeout.
        """
        calls = []

        def boom(*args, **kwargs):
            calls.append(1)
            raise OSError("no such model")

        monkeypatch.setitem(sys.modules, "sentence_transformers", _FakeSTModule(boom))
        for _ in range(3):
            with pytest.raises(RerankUnavailable):
                rerank_sync("q", ["doc"])
        assert len(calls) == 1


class TestScoring:
    def test_scores_are_returned_in_input_order(self, monkeypatch):
        import torch.nn as nn

        class FakeEncoder:
            activation_fn = nn.Sigmoid()

            def predict(self, pairs):
                # Deliberately unsorted, to catch an implementation that
                # reorders here instead of leaving that to the caller.
                return [0.1, 0.9, 0.5][: len(pairs)]

        monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: FakeEncoder())
        assert rerank_sync("q", ["a", "b", "c"]) == [0.1, 0.9, 0.5]

    def test_logit_outputs_are_normalized(self, monkeypatch):
        class FakeEncoder:
            def predict(self, pairs):
                return [-3.0, 6.0]

        monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: FakeEncoder())
        scores = rerank_sync("q", ["a", "b"])
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores[1] > scores[0]

    def test_out_of_range_activated_scores_are_clamped(self, monkeypatch):
        """A model that claims a sigmoid but returns 1.0000001 must not
        produce a score above the range RERANK_MIN_SCORE is expressed in."""
        import torch.nn as nn

        class FakeEncoder:
            activation_fn = nn.Sigmoid()

            def predict(self, pairs):
                return [1.0000001, -1e-9]

        monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: FakeEncoder())
        assert rerank_sync("q", ["a", "b"]) == [1.0, 0.0]
