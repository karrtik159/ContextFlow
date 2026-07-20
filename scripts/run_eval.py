"""
The single evaluation command — retrieval metrics and generation metrics.

    uv run python scripts/run_eval.py                  # retrieval only
    uv run python scripts/run_eval.py --generation     # + RAGAS
    uv run python scripts/run_eval.py --sweep-threshold

The two halves answer different questions and neither substitutes for the other:

  RETRIEVAL  (recall@k, MRR, nDCG@k) — did the right context come back at all?
             Needs PostgreSQL + pgvector and an embedding provider. Runs
             offline against a local sentence-transformer.

  GENERATION (RAGAS: faithfulness, answer relevance, context precision/recall)
             — given whatever context came back, is the answer grounded in it?
             Needs a RUNNING backend and a judge LLM, so it costs money and is
             opt-in.

A high faithfulness score over bad context still describes a useless system, so
retrieval metrics come first and always run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], label: str) -> int:
    print(f"\n{'#' * 72}\n# {label}\n{'#' * 72}")
    return subprocess.call([sys.executable, *cmd], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the ContextFlow evaluation suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--k", type=int, default=None, help="Cutoff for retrieval metrics.")
    parser.add_argument("--sweep-threshold", action="store_true",
                        help="Also sweep RETRIEVAL_MIN_SIMILARITY.")
    parser.add_argument("--generation", action="store_true",
                        help="Also run RAGAS. Requires a running backend and a judge LLM.")
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    retrieval_cmd = ["scripts/run_retrieval_eval.py"]
    if args.k is not None:
        retrieval_cmd += ["--k", str(args.k)]
    if args.sweep_threshold:
        retrieval_cmd.append("--sweep-threshold")
    if args.json_out:
        retrieval_cmd += ["--json", args.json_out]

    status = run(retrieval_cmd, "RETRIEVAL METRICS")
    if status != 0:
        print("\nretrieval eval failed; skipping generation metrics", file=sys.stderr)
        return status

    if args.generation:
        status = run(
            ["scripts/run_ragas_eval.py", "--api-base-url", args.api_base_url],
            "GENERATION METRICS (RAGAS)",
        )
    else:
        print(
            "\nGeneration metrics skipped. Pass --generation to run RAGAS "
            "(needs a running backend and a judge LLM)."
        )
    return status


if __name__ == "__main__":
    raise SystemExit(main())
