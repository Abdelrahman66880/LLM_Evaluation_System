"""Small synchronous client for OpenRouter's compatible API endpoints."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import Settings, get_settings


class OpenRouterClient:
    """Send JSON requests to OpenRouter with project-wide auth and headers."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, settings: Settings | None = None, *, timeout: float = 90) -> None:
        self.settings = settings or get_settings()
        self.api_key = self.settings.require_api_key()
        self.timeout = timeout

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            f"{self.BASE_URL}/{endpoint.lstrip('/')}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.settings.site_url,
                "X-Title": self.settings.app_name,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach OpenRouter: {exc.reason}") from exc
        if not isinstance(result, dict):
            raise RuntimeError("OpenRouter returned an unexpected response")
        return result

    def chat(self, messages: list[dict[str, str]], *, model: str | None = None) -> str:
        result = self.post(
            "chat/completions",
            {
                "model": model or self.settings.chat_model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 800,
            },
        )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("OpenRouter returned an unexpected chat completion") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenRouter returned an empty answer")
        return content.strip()
