"""Saved prompts with their settings (prompts.json next to the program)."""
from __future__ import annotations

import json
from datetime import datetime

from . import config

LIBRARY_FILE = config.ROOT / "prompts.json"


def load() -> list[dict]:
    try:
        data = json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [e for e in data if isinstance(e, dict) and e.get("name") and isinstance(e.get("ui"), dict)]


def _write(entries: list[dict]) -> None:
    tmp = LIBRARY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LIBRARY_FILE)


def save(name: str, ui: dict) -> list[dict]:
    """Add or overwrite an entry by name; newest first."""
    entries = [e for e in load() if e["name"] != name]
    entries.insert(0, {"name": name, "saved": datetime.now().isoformat(timespec="seconds"), "ui": ui})
    _write(entries)
    return entries


def delete(name: str) -> list[dict]:
    entries = [e for e in load() if e["name"] != name]
    _write(entries)
    return entries
