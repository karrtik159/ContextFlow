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

__all__ = [
    "DEFAULT_RRF_K",
    "FusedChunk",
    "RetrievalTrace",
    "RetrievedChunk",
    "SourceName",
    "StageRecord",
    "assign_citation_labels",
    "build_context_block",
    "reciprocal_rank_fusion",
    "run_retrieval",
]
