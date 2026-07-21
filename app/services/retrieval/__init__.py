"""Deterministic retrieval pipeline — see pipeline.py for the ordering."""

from app.services.retrieval.contracts import (
    RetrievalTrace,
    RetrievedChunk,
    SourceName,
    StageRecord,
)
from app.services.retrieval.fusion import (
    DEFAULT_RRF_K,
    FusedChunk,
    assign_citation_labels,
    reciprocal_rank_fusion,
)
from app.services.retrieval.pipeline import build_context_block, run_retrieval
from app.services.retrieval.rerank import rerank_fused
from app.services.retrieval.rewrite import (
    RewrittenQuery,
    build_sparse_query,
    deterministic_rewrite,
    extract_identifiers,
    rewrite_query,
)

__all__ = [
    "DEFAULT_RRF_K",
    "FusedChunk",
    "RetrievalTrace",
    "RetrievedChunk",
    "RewrittenQuery",
    "SourceName",
    "StageRecord",
    "assign_citation_labels",
    "build_context_block",
    "build_sparse_query",
    "deterministic_rewrite",
    "extract_identifiers",
    "reciprocal_rank_fusion",
    "rerank_fused",
    "rewrite_query",
    "run_retrieval",
]
