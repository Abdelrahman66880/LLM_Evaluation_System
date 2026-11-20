"""Question-answering agent that retrieves PDF context and calls OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

try:  # Support package imports and running ``python agent.py`` from RAG1/.
    from .objects import Answer
    from .prompts import system_agent_prompt
    from .tools import collecting_info
except ImportError:  # pragma: no cover - used for direct script execution
    from objects import Answer
    from prompts import system_agent_prompt
    from tools import collecting_info


CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_CHAT_MODEL = "openai/gpt-4o-mini"


def _generate_answer(question: str, context: str, model: str) -> str:
    """Ask an OpenRouter chat model to answer using only retrieved context."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add your OpenRouter key to the environment or .env file."
        )

    user_message = (
        f"Question:\n{question}\n\n"
        f"Retrieved PDF passages:\n{context or '[No relevant passages were retrieved.]'}"
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system_agent_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0,
            "max_tokens": 800,
        }
    ).encode("utf-8")
    request = Request(
        CHAT_COMPLETIONS_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_NAME", "LLM Evaluation System"),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter chat request failed ({exc.code}): {message}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter returned an unexpected chat completion response") from exc
    if isinstance(content, str) and content.strip():
        return content.strip()
    raise RuntimeError("OpenRouter returned an empty answer")


def answer_question(question: str, *, model: str | None = None) -> Answer:
    """Retrieve PDF passages and answer a question grounded in those passages.

    The first call builds the in-memory PDF index; later calls reuse it. The
    returned ``Answer.context`` contains the passages used to generate the answer.
    """
    load_dotenv()
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty")

    retrieved_context = collecting_info.invoke({"query": question})
    selected_model = model or os.getenv("OPENROUTER_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    answer = _generate_answer(question, retrieved_context, selected_model)
    return Answer(answer=answer, context=retrieved_context)