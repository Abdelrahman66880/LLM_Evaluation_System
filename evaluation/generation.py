"""Generate RAG predictions for a JSONL evaluation dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.agent import answer_question
from evaluation.io import read_jsonl, resolve_project_path


def generate_predictions(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
    model: str | None = None,
) -> Path:
    """Run the RAG pipeline on a dataset and save each completed record."""
    cases = read_jsonl(dataset_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        cases = cases[:limit]
    if not cases:
        raise ValueError("Evaluation dataset contains no cases")

    # Write as we go so a failed/interrupting request does not lose earlier work.
    destination = resolve_project_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for index, case in enumerate(cases, start=1):
            question = case.get("question")
            expected = case.get("expected_answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Dataset case {index} has no non-empty 'question'")
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError(f"Dataset case {index} has no non-empty 'expected_answer'")

            print(f"Generating answer {index}/{len(cases)}")
            result = answer_question(question, model=model)
            record: dict[str, Any] = {
                **case,
                "question": question,
                "expected_answer": expected,
                "actual_answer": result.answer,
                "context": result.context,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    return destination
