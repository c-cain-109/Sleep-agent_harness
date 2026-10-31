"""Small, dependency-free helpers for Sleep-Harness experiment memory.

This module does not modify the existing harness automatically. Integrate its
functions only after checking the project's current run_history.jsonl format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_RUN_FIELDS = {
    "run_id",
    "timestamp_utc",
    "status",
    "hypothesis",
    "changed_files",
    "decision",
}


def load_strategy(strategy_path: str | Path) -> str:
    """Read strategy.md; return an empty string only if it does not exist."""
    path = Path(strategy_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_run_history(history_path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL history. Raise on malformed lines to avoid silent data loss."""
    path = Path(history_path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            records.append(record)
    return records


def append_run_record(history_path: str | Path, record: dict[str, Any]) -> None:
    """Append one completed run record; intended to be called by the Harness."""
    missing = REQUIRED_RUN_FIELDS - record.keys()
    if missing:
        raise ValueError(f"Run record missing required fields: {sorted(missing)}")

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def build_agent_memory_context(
    strategy_path: str | Path,
    history_path: str | Path,
    recent_runs: int = 5,
) -> str:
    """Build a prompt section from strategy summary and recent factual runs."""
    if recent_runs < 1:
        raise ValueError("recent_runs must be >= 1")

    strategy = load_strategy(strategy_path)
    records = load_run_history(history_path)
    recent = records[-recent_runs:]

    sections = [
        "# Persistent Experiment Memory",
        "Use this memory to avoid repeating prior attempts.",
        "The run-history records are factual outputs from the Harness. Do not alter or invent them.",
        "",
        "## Strategy summary",
        strategy or "(No strategy.md exists yet.)",
        "",
        f"## Most recent {len(recent)} run records",
    ]

    if not recent:
        sections.append("(No experiment records yet.)")
    else:
        for record in recent:
            sections.append(json.dumps(record, ensure_ascii=False, sort_keys=True))

    return "\n".join(sections)
