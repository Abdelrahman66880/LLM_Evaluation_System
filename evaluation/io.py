"""Shared JSONL input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative data paths from the repository root, not the shell cwd."""
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read object records from JSON Lines with useful line-number errors."""
    file_path = resolve_project_path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL input not found: {file_path}")
    records: list[dict[str, Any]] = []
    with file_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {file_path}, line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {file_path}, line {line_number}")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    """Write JSON records one per line, creating the destination directory."""
    file_path = resolve_project_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return file_path
