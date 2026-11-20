"""Score RAG prediction records with DeepEval and an OpenRouter judge."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase, SingleTurnParams

from config import get_settings
from evaluation.io import read_jsonl, write_jsonl

METRIC_THRESHOLD = 0.7


def score_predictions(
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    model: str | None = None,
    limit: int | None = None,
) -> Path:
    """Evaluate saved predictions and write metric scores beside each record."""
    records = read_jsonl(predictions_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        records = records[:limit]
    if not records:
        raise ValueError("Prediction file contains no records")

    test_cases: list[LLMTestCase] = []
    for index, record in enumerate(records, start=1):
        for field in ("question", "actual_answer", "expected_answer", "context"):
            if field not in record:
                raise ValueError(f"Prediction record {index} is missing '{field}'")
        context = record["context"]
        retrieval_context = [context] if isinstance(context, str) else context
        if not isinstance(retrieval_context, list) or not all(
            isinstance(item, str) for item in retrieval_context
        ):
            raise ValueError(f"Prediction record {index} has an invalid context")
        test_cases.append(
            LLMTestCase(
                input=record["question"],
                actual_output=record["actual_answer"],
                expected_output=record["expected_answer"],
                retrieval_context=retrieval_context,
            )
        )

    settings = get_settings()
    judge = OpenRouterModel(
        model=model or os.getenv("OPENROUTER_EVAL_MODEL", settings.chat_model),
        api_key=settings.require_api_key(),
        base_url="https://openrouter.ai/api/v1",
    )
    metrics = [
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge, async_mode=False),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=judge, async_mode=False),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=judge, async_mode=False),
        GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            threshold=METRIC_THRESHOLD,
            model=judge,
            async_mode=False,
        ),
    ]

    result = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        # DeepEval's async runner calls a_measure() even for async_mode=False
        # metrics. Use the synchronous runner to avoid burst credit reservations.
        async_config=AsyncConfig(run_async=False, max_concurrent=1),
        cache_config=CacheConfig(write_cache=True, use_cache=True),
    )

    scored_records: list[dict[str, Any]] = []
    for original, test_result in zip(records, result.test_results, strict=True):
        scores: dict[str, float | None] = {
            "answer_relevancy": None,
            "faithfulness": None,
            "context_relevancy": None,
            "correctness": None,
        }
        for metric_data in test_result.metrics_data or []:
            name = metric_data.name.strip().lower()
            if name == "answer relevancy":
                scores["answer_relevancy"] = metric_data.score
            elif name == "faithfulness":
                scores["faithfulness"] = metric_data.score
            elif name == "contextual relevancy":
                scores["context_relevancy"] = metric_data.score
            elif name.startswith("correctness"):
                scores["correctness"] = metric_data.score
        scored_records.append({**original, **scores})

    return write_jsonl(output_path, scored_records)
