"""
Retrieval evaluation — the single command that reports retrieval metrics.

    uv run python scripts/run_retrieval_eval.py
    uv run python scripts/run_retrieval_eval.py --sweep-threshold
    uv run python scripts/run_retrieval_eval.py --json results.json

Needs a reachable PostgreSQL with pgvector and a working embedding provider.
It ingests the golden corpus into a scratch schema, runs the real retrieval
pipeline against it, and scores the results — so it measures the pipeline, not
a reimplementation of it.

Set EMBEDDING_PROVIDER=huggingface to run entirely locally with no API spend.
Remember that EMBEDDING_DIMENSIONS must match the provider and the pgvector
column, so a local run wants its own database (see --help).

The graph and memory arms are stubbed out: the golden set contains no graph
edges or user memories, so leaving them live would only add latency and noise.
Retrieval quality here is the dense corpus arm plus fusion plus thresholding.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.evals.golden_set import load_golden_set, resolve_relevant_chunk_ids  # noqa: E402
from app.evals.retrieval_metrics import aggregate, score_query  # noqa: E402
from app.models.document import Chunk  # noqa: E402
from app.models.user import User  # noqa: E402


async def _stub_arm(**kwargs):
    return []


def _install_arm_stubs() -> None:
    from app.services.retrieval import pipeline as pipeline_mod

    pipeline_mod.search_graph = _stub_arm
    pipeline_mod.search_memory = _stub_arm


async def build_corpus(Session, golden, user_id: uuid.UUID) -> dict[str, list[tuple[uuid.UUID, str]]]:
    """Ingest the golden documents and return document key -> [(chunk_id, text)]."""
    from app.services.ingestion import ingest_document

    chunks_by_document: dict[str, list[tuple[uuid.UUID, str]]] = {}

    async with Session() as db:
        db.add(
            User(
                id=user_id,
                email=f"{user_id}@eval.local",
                username=str(user_id)[:12],
                hashed_password="x",
            )
        )
        await db.commit()

    for doc in golden.documents:
        async with Session() as db:
            result = await ingest_document(
                db, user_id=user_id, text=doc.text, title=doc.title,
                source_uri=f"golden://{doc.key}",
            )
            await db.commit()
        async with Session() as db:
            rows = (
                await db.execute(
                    select(Chunk.id, Chunk.text)
                    .where(Chunk.document_id == result.document_id, Chunk.user_id == user_id)
                    .order_by(Chunk.chunk_index)
                )
            ).all()
        chunks_by_document[doc.key] = [(r[0], r[1]) for r in rows]

    return chunks_by_document


@contextlib.contextmanager
def override(**values):
    """Temporarily set settings, restoring them even on failure.

    The ablation below flips SPARSE_ENABLED and RERANK_ENABLED between runs. A
    leaked override would make every subsequent configuration measure the
    previous one's flags and quietly report a false comparison.
    """
    previous = {name: getattr(settings, name) for name in values}
    for name, value in values.items():
        setattr(settings, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(settings, name, value)


async def evaluate(
    Session,
    golden,
    user_id,
    chunks_by_document,
    *,
    k: int,
    min_similarity: float,
):
    """Run the real pipeline for every answerable query and score it."""
    from app.services.embeddings import embed_text_async
    from app.services.retrieval.pipeline import run_retrieval

    original_floor = settings.RETRIEVAL_MIN_SIMILARITY
    settings.RETRIEVAL_MIN_SIMILARITY = min_similarity
    try:
        scores = []
        for query in golden.answerable:
            relevant = resolve_relevant_chunk_ids(
                query.relevant, chunks_by_document=chunks_by_document
            )
            embedding = await embed_text_async(query.query)
            async with Session() as db:
                trace = await run_retrieval(
                    db,
                    user_id=str(user_id),
                    original_query=query.query,
                    retrieval_query=query.query,
                    query_embedding=embedding,
                    top_k=k,
                )
            retrieved = [str(c.chunk_id) for c in trace.final_chunks if c.chunk_id]
            scores.append(
                score_query(query=query.query, retrieved=retrieved, relevant=relevant, k=k)
            )

        # Unanswerable queries are scored on abstention, not on recall.
        abstentions = []
        for query in golden.unanswerable:
            embedding = await embed_text_async(query.query)
            async with Session() as db:
                trace = await run_retrieval(
                    db,
                    user_id=str(user_id),
                    original_query=query.query,
                    retrieval_query=query.query,
                    query_embedding=embedding,
                    top_k=k,
                )
            abstentions.append(
                {
                    "query": query.query,
                    "category": query.category,
                    "terms_all_in_corpus": query.terms_all_in_corpus,
                    "retrieved": len(trace.final_chunks),
                    "abstained": not trace.has_context,
                    "top_score": (
                        trace.final_chunks[0].score if trace.final_chunks else None
                    ),
                }
            )
        return scores, abstentions
    finally:
        settings.RETRIEVAL_MIN_SIMILARITY = original_floor


def print_report(scores, abstentions, k: int, min_similarity: float) -> dict:
    agg = aggregate(scores, k=k)
    print(f"\n{'=' * 72}")
    print(f"RETRIEVAL METRICS   k={k}  min_similarity={min_similarity}  "
          f"model={settings.EMBEDDING_MODEL}")
    print("=" * 72)
    for key in (f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}", f"hit_rate@{k}"):
        print(f"  {key:<16} {agg[key]:.3f}")
    print(f"  {'queries':<16} {agg['queries']}")
    print(f"  {'total misses':<16} {agg['total_misses']}")

    misses = [s for s in scores if s.missed_everything]
    if misses:
        print(f"\n  Queries that retrieved NOTHING relevant ({len(misses)}):")
        for s in misses:
            print(f"    - {s.query}")

    weak = [s for s in scores if not s.missed_everything and s.mrr < 0.5]
    if weak:
        print(f"\n  Relevant chunk found, but ranked 3rd or worse ({len(weak)}):")
        for s in weak:
            print(f"    - (rr={s.mrr:.2f}) {s.query}")

    if abstentions:
        correct = sum(1 for a in abstentions if a["abstained"])
        print(f"\n  Abstention on unanswerable queries: {correct}/{len(abstentions)}")

        # Per-category, because abstention accuracy is not one number. A
        # mechanism can be perfect on out-of-database requests and useless on
        # underspecified ones, and the aggregate hides precisely that.
        by_category: dict[str, list] = {}
        for a in abstentions:
            by_category.setdefault(a.get("category") or "uncategorised", []).append(a)
        print("\n    by category:")
        for category, items in sorted(by_category.items()):
            ok = sum(1 for i in items if i["abstained"])
            print(f"      {category:<22} {ok}/{len(items)}")

        # The split that decides whether a term-coverage heuristic is doing
        # real work or just detecting a word the author left out.
        hard = [a for a in abstentions if a.get("terms_all_in_corpus")]
        easy = [a for a in abstentions if not a.get("terms_all_in_corpus")]
        if hard:
            ok_hard = sum(1 for a in hard if a["abstained"])
            ok_easy = sum(1 for a in easy if a["abstained"])
            print(
                f"\n    all query terms present in corpus: {ok_hard}/{len(hard)}  "
                f"(a coverage heuristic is structurally blind here)"
            )
            print(f"    some query term absent:            {ok_easy}/{len(easy)}")

        print()
        for a in abstentions:
            mark = "OK  " if a["abstained"] else "LEAK"
            top = f"{a['top_score']:.3f}" if a["top_score"] is not None else "n/a"
            hard_flag = "H" if a.get("terms_all_in_corpus") else " "
            print(
                f"    [{mark}]{hard_flag} {(a.get('category') or '?'):<20} "
                f"top={top}  {a['query'][:70]}"
            )

    return {"aggregate": agg, "abstentions": abstentions}


# The four configurations that answer "did Phase 3 help?". Each isolates one
# addition, because "dense + sparse + rerank beats dense" does not say which of
# the two additions did the work — or whether one of them hurt and the other
# more than compensated.
ABLATIONS = [
    ("dense only (Phase 2)", {"SPARSE_ENABLED": False, "RERANK_ENABLED": False}),
    ("+ sparse", {"SPARSE_ENABLED": True, "RERANK_ENABLED": False}),
    ("+ rerank", {"SPARSE_ENABLED": False, "RERANK_ENABLED": True}),
    ("+ both (Phase 3)", {"SPARSE_ENABLED": True, "RERANK_ENABLED": True}),
]


async def ablate(Session, golden, user_id, chunks_by_document, *, k: int, min_similarity: float):
    """Measure each Phase 3 addition in isolation.

    This is the Phase 3 exit criterion: 'measurable recall@k and nDCG
    improvement on a labelled set'. A single number from the final
    configuration cannot meet it, because there is nothing to compare it to.
    """
    print(f"\n{'=' * 88}")
    print(f"ABLATION — k={k}  min_similarity={min_similarity}  model={settings.EMBEDDING_MODEL}")
    print("=" * 88)
    header = (
        f"{'configuration':<22} {'recall':>8} {'ndcg':>8} {'mrr':>8} "
        f"{'prec':>8} {'misses':>7} {'abstain':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for label, flags in ABLATIONS:
        with override(**flags):
            scores, abstentions = await evaluate(
                Session, golden, user_id, chunks_by_document,
                k=k, min_similarity=min_similarity,
            )
        agg = aggregate(scores, k=k)
        abstained = sum(1 for a in abstentions if a["abstained"])
        rows.append(
            {
                "configuration": label,
                **flags,
                "recall": agg[f"recall@{k}"],
                "ndcg": agg[f"ndcg@{k}"],
                "mrr": agg["mrr"],
                "precision": agg[f"precision@{k}"],
                "misses": agg["total_misses"],
                "abstained": abstained,
                "abstain_total": len(abstentions),
            }
        )
        print(
            f"{label:<22} {agg[f'recall@{k}']:>8.3f} {agg[f'ndcg@{k}']:>8.3f} "
            f"{agg['mrr']:>8.3f} {agg[f'precision@{k}']:>8.3f} "
            f"{agg['total_misses']:>7d} {abstained:>5d}/{len(abstentions)}"
        )

    baseline, final = rows[0], rows[-1]
    print(
        f"\n  Phase 2 -> Phase 3:  recall {baseline['recall']:+.3f} -> "
        f"{final['recall']:.3f} ({final['recall'] - baseline['recall']:+.3f})   "
        f"nDCG {baseline['ndcg']:.3f} -> {final['ndcg']:.3f} "
        f"({final['ndcg'] - baseline['ndcg']:+.3f})"
    )
    print(
        "  A delta of 0.000 on every metric means the golden set still cannot "
        "discriminate,\n  not that the change is neutral. Read the per-query "
        "misses before concluding either."
    )
    return rows


async def sweep_rerank(Session, golden, user_id, chunks_by_document, k: int, min_similarity: float):
    """Calibrate RERANK_MIN_SCORE — the abstention dial Phase 5 could not find.

    Phase 5's threshold sweep showed no cosine floor both preserves recall and
    rejects unanswerable queries. The cross-encoder score is a different signal;
    this sweep is the test of whether it separates them where cosine could not.
    """
    print(f"\n{'=' * 72}")
    print("RERANK SCORE SWEEP — RERANK_MIN_SCORE")
    print("=" * 72)
    header = f"{'floor':>7} {'recall':>8} {'ndcg':>8} {'mrr':>8} {'misses':>7} {'abstain':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for floor in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        with override(SPARSE_ENABLED=True, RERANK_ENABLED=True, RERANK_MIN_SCORE=floor):
            scores, abstentions = await evaluate(
                Session, golden, user_id, chunks_by_document,
                k=k, min_similarity=min_similarity,
            )
        agg = aggregate(scores, k=k)
        abstained = sum(1 for a in abstentions if a["abstained"])
        rows.append(
            {
                "rerank_min_score": floor,
                "recall": agg[f"recall@{k}"],
                "ndcg": agg[f"ndcg@{k}"],
                "mrr": agg["mrr"],
                "misses": agg["total_misses"],
                "abstained": abstained,
                "abstain_total": len(abstentions),
            }
        )
        print(
            f"{floor:>7.2f} {agg[f'recall@{k}']:>8.3f} {agg[f'ndcg@{k}']:>8.3f} "
            f"{agg['mrr']:>8.3f} {agg['total_misses']:>7d} "
            f"{abstained:>5d}/{len(abstentions)}"
        )
    return rows


async def sweep_threshold(Session, golden, user_id, chunks_by_document, k: int):
    """Calibrate RETRIEVAL_MIN_SIMILARITY against the golden set.

    This number decides between the honest-empty path and answering from noise,
    and it was set to 0.25 by guesswork. A sweep is the only way to choose it
    that is not vibes: too low and unanswerable queries retrieve noise, too high
    and answerable queries retrieve nothing.
    """
    print(f"\n{'=' * 72}")
    print("THRESHOLD SWEEP — RETRIEVAL_MIN_SIMILARITY")
    print("=" * 72)
    header = f"{'floor':>7} {'recall':>8} {'ndcg':>8} {'mrr':>8} {'misses':>7} {'abstain':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for floor in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
        scores, abstentions = await evaluate(
            Session, golden, user_id, chunks_by_document, k=k, min_similarity=floor
        )
        agg = aggregate(scores, k=k)
        abstained = sum(1 for a in abstentions if a["abstained"])
        rows.append(
            {
                "min_similarity": floor,
                "recall": agg[f"recall@{k}"],
                "ndcg": agg[f"ndcg@{k}"],
                "mrr": agg["mrr"],
                "misses": agg["total_misses"],
                "abstained": abstained,
                "abstain_total": len(abstentions),
            }
        )
        print(
            f"{floor:>7.2f} {agg[f'recall@{k}']:>8.3f} {agg[f'ndcg@{k}']:>8.3f} "
            f"{agg['mrr']:>8.3f} {agg['total_misses']:>7d} "
            f"{abstained:>4d}/{len(abstentions)}"
        )

    best = max(rows, key=lambda r: (r["abstained"], r["recall"]))
    print(
        f"\n  Highest floor that keeps full recall while abstaining correctly is a "
        f"judgement call, not an argmax — read the table.\n"
        f"  Best abstention with best recall in this run: "
        f"min_similarity={best['min_similarity']} "
        f"(recall={best['recall']:.3f}, abstained={best['abstained']}/{best['abstain_total']})"
    )
    return rows


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against the committed golden set.",
        epilog=(
            "Local, no-API-spend run:\n"
            "  EMBEDDING_PROVIDER=huggingface EMBEDDING_MODEL=all-MiniLM-L6-v2 \\\n"
            "  EMBEDDING_DIMENSIONS=384 EMBEDDING_MAX_TOKENS=256 \\\n"
            "  CHUNK_TARGET_TOKENS=192 CHUNK_OVERLAP_TOKENS=32 \\\n"
            "  uv run python scripts/run_retrieval_eval.py\n"
            "(the pgvector column width is built from EMBEDDING_DIMENSIONS at "
            "migration time, so a 384-dim run needs its own database)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--k", type=int, default=settings.RETRIEVAL_TOP_K)
    parser.add_argument("--min-similarity", type=float, default=settings.RETRIEVAL_MIN_SIMILARITY)
    parser.add_argument("--sweep-threshold", action="store_true")
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="measure dense / +sparse / +rerank / both — the Phase 3 exit criterion",
    )
    parser.add_argument(
        "--sufficiency",
        action="store_true",
        help=(
            "enable the SUFFICIENCY gate. Read the false-abstention count on "
            "ANSWERABLE queries, not just the abstention count — refusing a real "
            "question is the worse error."
        ),
    )
    parser.add_argument(
        "--sufficiency-max-uncovered",
        type=int,
        default=None,
        help="override SUFFICIENCY_MAX_UNCOVERED_TERMS",
    )
    parser.add_argument(
        "--sweep-rerank",
        action="store_true",
        help="calibrate RERANK_MIN_SCORE, the abstention dial cosine could not provide",
    )
    parser.add_argument("--golden-set", default="tests/test_eval/fixtures/golden_set.json")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    _install_arm_stubs()

    if args.sufficiency:
        settings.SUFFICIENCY_ENABLED = True
    if args.sufficiency_max_uncovered is not None:
        settings.SUFFICIENCY_MAX_UNCOVERED_TERMS = args.sufficiency_max_uncovered
    print(
        f"sufficiency gate: {'ON' if settings.SUFFICIENCY_ENABLED else 'off'} "
        f"(max_uncovered={settings.SUFFICIENCY_MAX_UNCOVERED_TERMS})"
    )

    golden = load_golden_set(args.golden_set)
    print(
        f"golden set: {len(golden.documents)} documents, "
        f"{len(golden.answerable)} answerable queries, "
        f"{len(golden.unanswerable)} unanswerable"
    )

    engine = create_async_engine(settings.database_url)
    Session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    user_id = uuid.uuid4()

    try:
        chunks_by_document = await build_corpus(Session, golden, user_id)
        total_chunks = sum(len(v) for v in chunks_by_document.values())
        print(f"ingested {total_chunks} chunks across {len(chunks_by_document)} documents")

        payload: dict = {
            "embedding_model": settings.EMBEDDING_MODEL,
            "chunk_target_tokens": settings.CHUNK_TARGET_TOKENS,
            "k": args.k,
        }

        scores, abstentions = await evaluate(
            Session, golden, user_id, chunks_by_document,
            k=args.k, min_similarity=args.min_similarity,
        )
        payload.update(print_report(scores, abstentions, args.k, args.min_similarity))

        if args.ablate:
            payload["ablation"] = await ablate(
                Session, golden, user_id, chunks_by_document,
                k=args.k, min_similarity=args.min_similarity,
            )

        if args.sweep_rerank:
            payload["rerank_sweep"] = await sweep_rerank(
                Session, golden, user_id, chunks_by_document,
                args.k, args.min_similarity,
            )

        if args.sweep_threshold:
            payload["sweep"] = await sweep_threshold(
                Session, golden, user_id, chunks_by_document, args.k
            )

        if args.json_out:
            Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nwrote {args.json_out}")
    finally:
        await engine.dispose()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
