"""
Contracts shared by every retrieval stage.

`rank` is mandatory on every retrieved item and is the reason this module
exists. Reciprocal Rank Fusion operates on ranks, not scores — and the previous
design had neither: two of the three retrieval tools returned prose with no
scores at all, while a YAML prompt instructed the model to "combine all results
using Reciprocal Rank Fusion". The model was asked to execute a ranking
algorithm over data it was never given, and the output looked fused and was
unfalsifiable. Ranks are now produced by the code that does the ranking.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

# "bm25" is reserved for the sparse arm added in Phase 3; declaring it now keeps
# that from being a contract change.
SourceName = Literal["vector", "bm25", "messages", "graph", "memory"]


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved item, from any source.

    `score` is the RAW score from its own source and is NOT comparable across
    sources — a cosine similarity, a graph hop count, and a Mem0 relevance score
    share no scale. That incomparability is precisely why fusion works on
    `rank`, which is meaningful within a list, rather than on `score`.
    """

    text: str
    source: SourceName
    rank: int  # 1-based, WITHIN this item's own source list
    score: float
    chunk_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None
    heading_path: str | None = None
    # Assigned at context assembly ("[1]"), not at retrieval — a chunk's label
    # depends on the final ordering, which fusion has not run yet.
    citation_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> str:
        """Identity for fusion — deliberately source-INDEPENDENT.

        Corpus chunks dedup by id. Graph and memory hits have no stable id, so
        they fall back to normalized text.

        The source is NOT part of the key. Qualifying id-less items by source
        would mean the same fact surfaced by two different arms never fuses,
        which removes exactly the cross-source agreement that RRF exists to
        reward — the graph and memory arms would each be scored in isolation
        and fusion would degrade to a weighted concatenation.
        """
        if self.chunk_id is not None:
            return f"chunk:{self.chunk_id}"
        return f"text:{' '.join(self.text.split()).casefold()[:512]}"


@dataclass
class StageRecord:
    """One stage of the pipeline, as it actually ran.

    Persisted. This is the product, not a debug afterthought — it is what makes
    retrieval quality measurable instead of anecdotal, and Phase 5's recall@k
    and nDCG are computed from it.
    """

    name: str
    latency_ms: int
    input_summary: str
    output_summary: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "latency_ms": self.latency_ms,
            "input": self.input_summary,
            "output": self.output_summary,
            "meta": self.metadata,
        }


@dataclass
class RetrievalTrace:
    """The full record of one retrieval run."""

    trace_id: uuid.UUID
    user_id: str
    original_query: str
    normalized_query: str
    rewritten_query: str | None = None
    stages: list[StageRecord] = field(default_factory=list)
    final_chunks: list[RetrievedChunk] = field(default_factory=list)
    total_latency_ms: int = 0

    def record(self, stage: StageRecord) -> None:
        self.stages.append(stage)

    @property
    def has_context(self) -> bool:
        """False means the honest 'no context' path, not an error.

        Nothing clearing the relevance floor is a legitimate outcome. The
        previous implementation had no floor at all and rendered whatever came
        back as "Found 5 relevant results:", so an off-topic query handed the
        synthesizer five pieces of noise labelled relevant.
        """
        return bool(self.final_chunks)

    def stages_as_json(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.stages]
