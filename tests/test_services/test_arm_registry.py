"""
Arm registry contract (Phase 8).

The registry turned three synchronized edits (fan-out list, weights dict,
literal) into one declaration — these tests are what keeps a declaration
honest. They enforce what was previously only convention: every arm has a
resolvable seam on the pipeline module, a positive weight, a valid kind, and
a build function that produces a working factory.
"""

from __future__ import annotations

import typing
import uuid

import pytest

from app.core.config import settings
from app.services.retrieval import pipeline as pipeline_mod
from app.services.retrieval.arms import ARM_REGISTRY, ArmContext, source_weights
from app.services.retrieval.contracts import SourceName


def _ctx() -> ArmContext:
    return ArmContext(
        user_id="user-1",
        scoped_uuid=uuid.uuid4(),
        retrieval_query="how do log files rotate",
        sparse_query="how do log files rotate",
        query_embedding=[0.1] * 8,
    )


def test_registry_names_are_unique():
    names = [spec.name for spec in ARM_REGISTRY]
    assert len(names) == len(set(names))


def test_registry_names_are_in_the_source_literal():
    """`source` on every RetrievedChunk is typed as SourceName; an arm whose
    name is not in the literal would stamp chunks the contract cannot type."""
    allowed = set(typing.get_args(SourceName))
    for spec in ARM_REGISTRY:
        assert spec.name in allowed, f"arm {spec.name!r} missing from SourceName"


def test_registry_kinds_are_valid():
    for spec in ARM_REGISTRY:
        assert spec.kind in ("db", "external")


def test_registry_weights_are_positive():
    for spec in ARM_REGISTRY:
        assert spec.weight > 0, f"arm {spec.name!r} has non-positive weight"


def test_pipeline_weights_derive_from_the_registry():
    assert pipeline_mod.SOURCE_WEIGHTS == source_weights()
    assert set(pipeline_mod.SOURCE_WEIGHTS) == {spec.name for spec in ARM_REGISTRY}


def test_fn_names_resolve_on_the_pipeline_module():
    """The patch seam: every fn_name must be an attribute of the pipeline
    module, because that is where the stage tests stub arms. A registry entry
    whose seam does not resolve is an arm that silently never runs."""
    for spec in ARM_REGISTRY:
        fn = getattr(pipeline_mod, spec.fn_name, None)
        assert callable(fn), f"pipeline.{spec.fn_name} missing for arm {spec.name!r}"


@pytest.mark.asyncio
async def test_build_produces_a_factory_that_calls_the_given_fn():
    """`build(ctx, fn)` must call the fn it was handed — not a private import.

    This is the property that keeps monkeypatched stubs effective: the
    pipeline resolves fn on its own namespace and passes it in, so a build
    function importing the arm directly would bypass every test stub.
    """
    ctx = _ctx()
    for spec in ARM_REGISTRY:
        seen: dict = {}

        async def fake_fn(*args, _seen=seen, **kwargs):
            _seen["args"] = args
            _seen["kwargs"] = kwargs
            return []

        factory = spec.build(ctx, fake_fn)
        result = await factory(None)

        assert result == [], f"arm {spec.name!r} factory did not return the fn result"
        assert "kwargs" in seen, f"arm {spec.name!r} factory never called its fn"
        if spec.kind == "db":
            # DB arms receive the session positionally; scope is the UUID.
            assert seen["kwargs"].get("user_id") == ctx.scoped_uuid
        else:
            # External arms take no session; scope is the string user_id.
            assert seen["kwargs"].get("user_id") == ctx.user_id


def test_sparse_arm_honours_its_flag(monkeypatch):
    spec = next(s for s in ARM_REGISTRY if s.name == "bm25")

    monkeypatch.setattr(settings, "SPARSE_ENABLED", True)
    assert spec.enabled() is True

    monkeypatch.setattr(settings, "SPARSE_ENABLED", False)
    assert spec.enabled() is False


def test_always_on_arms_have_no_flag():
    for spec in ARM_REGISTRY:
        if spec.name == "bm25":
            continue
        assert spec.enabled() is True
