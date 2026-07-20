"""
RAGAS evaluation framework for the ContextFlow RAG pipeline.

This module is designed to work in two phases:
1. Answer-level evaluation now, using `/api/v1/rag/query`.
2. Retrieval-level evaluation later, once the backend exposes retrieved contexts
   in an eval-friendly response mode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import HuggingFaceEmbeddings, LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import Metric
from ragas.metrics.collections import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from app.core.config import settings
from app.services.llm_provider import get_openai_compatible_kwargs

METRIC_NAME_ORDER = [
    "answer_relevancy",
    "answer_correctness",
    "faithfulness",
    "context_precision",
    "context_recall",
]


class EvalCase(BaseModel):
    user_input: str = Field(min_length=1)
    user_id: str = Field(default="anonymous")
    response: str | None = None
    reference: str | None = None
    retrieved_contexts: list[str] | None = None
    reference_contexts: list[str] | None = None

    def to_ragas_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "user_input": self.user_input,
            "response": self.response,
            "reference": self.reference,
            "retrieved_contexts": self.retrieved_contexts,
            "reference_contexts": self.reference_contexts,
        }
        return {key: value for key, value in record.items() if value is not None}


class FastAPIRagTarget:
    def __init__(self, api_base_url: str, timeout_seconds: float = 45.0):
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def answer(self, user_input: str, user_id: str) -> str:
        response = self._request_json(
            "POST",
            "/api/v1/rag/query",
            {"query": user_input, "user_id": user_id},
        )
        payload = response.get("data", {})
        answer = payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("RAG endpoint returned no answer field")
        return answer

    def _request_json(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        # /rag/query has no unscoped mode: anonymous callers get 401, and an
        # anonymous caller supplying a user_id gets 403. The eval harness drives
        # arbitrary per-case user_ids, so it authenticates as an internal
        # service — the one caller class permitted to name its own scope.
        token = settings.RAG_SERVICE_TOKEN.get_secret_value()
        if not token:
            raise RuntimeError(
                "RAG_SERVICE_TOKEN is not configured. The eval target authenticates as an "
                "internal service; without it every case fails with 401."
            )
        with httpx.Client(base_url=self.api_base_url, timeout=self.timeout_seconds) as client:
            response = client.request(method, path, json=payload, headers={"X-RAG-Service-Token": token})
            response.raise_for_status()
            return {"data": response.json()}


def load_eval_cases(path: str | Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Eval dataset must be a JSON array of cases")
    return [EvalCase.model_validate(item) for item in raw]


def materialize_responses(cases: list[EvalCase], target: FastAPIRagTarget) -> list[EvalCase]:
    hydrated: list[EvalCase] = []
    for case in cases:
        if case.response:
            hydrated.append(case)
            continue

        hydrated.append(
            case.model_copy(
                update={"response": target.answer(case.user_input, case.user_id)},
            )
        )
    return hydrated


def build_evaluation_dataset(cases: list[EvalCase]) -> EvaluationDataset:
    return EvaluationDataset.from_list([case.to_ragas_record() for case in cases])


def select_metric_names(cases: list[EvalCase]) -> list[str]:
    has_reference = all(case.reference for case in cases)
    has_retrieved_contexts = all(case.retrieved_contexts for case in cases)

    metric_names: list[str] = ["answer_relevancy"]

    if has_reference:
        metric_names.append("answer_correctness")

    if has_retrieved_contexts:
        metric_names.append("faithfulness")

    if has_reference and has_retrieved_contexts:
        metric_names.extend(["context_precision", "context_recall"])

    return metric_names


def build_metrics(
    metric_names: list[str],
    *,
    llm: LangchainLLMWrapper,
    embeddings: Any,
) -> list[Metric]:
    metric_registry: dict[str, Metric] = {
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=embeddings),
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }
    return [metric_registry[name] for name in metric_names]


def build_judges() -> tuple[LangchainLLMWrapper, Any]:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=os.getenv("RAGAS_EVAL_LLM", settings.LLM_MODEL),
            temperature=0,
            **get_openai_compatible_kwargs(),
        )
    )

    embedding_model = os.getenv("RAGAS_EVAL_EMBEDDING_MODEL", settings.EMBEDDING_MODEL)
    if settings.EMBEDDING_PROVIDER == "huggingface":
        embeddings = HuggingFaceEmbeddings(model=embedding_model)
        return llm, embeddings

    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model, **get_openai_compatible_kwargs()))
    return llm, embeddings


def evaluate_cases(cases: list[EvalCase]) -> Any:
    dataset = build_evaluation_dataset(cases)
    metric_names = select_metric_names(cases)
    llm, embeddings = build_judges()
    metrics = build_metrics(metric_names, llm=llm, embeddings=embeddings)
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        experiment_name="contextflow_dev-ragas",
        show_progress=True,
        raise_exceptions=False,
    )


def summarise_result(result: Any) -> dict[str, Any]:
    frame = result.to_pandas()
    summary: dict[str, Any] = {
        "rows": int(len(frame)),
        "metrics": {},
    }
    for column in frame.columns:
        if column in {"user_input", "response", "reference", "retrieved_contexts", "reference_contexts"}:
            continue
        if frame[column].dtype.kind not in {"i", "f"}:
            continue
        summary["metrics"][column] = float(frame[column].mean())
    return summary
