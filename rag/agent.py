"""Retrieval-augmented question answering."""

from __future__ import annotations

from config import get_settings
from providers.openrouter import OpenRouterClient
from rag.prompts import SYSTEM_PROMPT
from rag.retrieval import PDFRetriever, get_default_retriever
from schemas import Answer


def answer_question(
    question: str,
    *,
    retriever: PDFRetriever | None = None,
    model: str | None = None,
) -> Answer:
    """Retrieve relevant PDF text and ask OpenRouter for a grounded answer."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    context = (retriever or get_default_retriever()).retrieve(question)
    client = OpenRouterClient()
    answer = client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved PDF passages:\n{context}",
            },
        ],
        model=model or get_settings().chat_model,
    )
    return Answer(answer=answer, context=context)
