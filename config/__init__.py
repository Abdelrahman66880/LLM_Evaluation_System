"""Application configuration and project paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _find_project_root() -> Path:
    configured = os.getenv("LLM_EVALUATION_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    source = Path(__file__).resolve()
    candidates = [*source.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _configured_path(variable: str, default: Path) -> Path:
    value = Path(os.getenv(variable, str(default))).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


@dataclass(frozen=True)
class Settings:
    """Runtime settings sourced from environment variables and the project .env."""

    openrouter_api_key: str | None
    chat_model: str
    embedding_model: str
    pdf_path: Path
    index_path: Path
    site_url: str
    app_name: str

    @classmethod
    def from_environment(cls) -> Settings:
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            chat_model=os.getenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini"),
            embedding_model=os.getenv(
                "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
            ),
            pdf_path=_configured_path("RAG_PDF_PATH", PROJECT_ROOT / "PDFs"),
            index_path=_configured_path(
                "RAG_INDEX_PATH", PROJECT_ROOT / ".data" / "vector_index.json"
            ),
            site_url=os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            app_name=os.getenv("OPENROUTER_APP_NAME", "LLM Evaluation System"),
        )

    def require_api_key(self) -> str:
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Add your OpenRouter key to the project .env file."
            )
        return self.openrouter_api_key


def get_settings() -> Settings:
    """Return settings after loading the project-level .env file."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings.from_environment()
