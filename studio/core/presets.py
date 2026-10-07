"""Style presets: prompt prefix/suffix, negative prompt and recommended settings."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config


@dataclass
class Preset:
    name: str
    note: str = ""
    prefix: str = ""
    suffix: str = ""
    negative: str = ""
    settings: dict = field(default_factory=dict)

    def build(self, prompt: str, negative: str) -> tuple[str, str]:
        prompt = f"{self.prefix}{prompt.strip()}{self.suffix}".strip(" ,")
        negative = ", ".join(p for p in (self.negative.strip(), negative.strip()) if p)
        return prompt, negative


def load_presets() -> list[Preset]:
    try:
        data = json.loads(config.PRESETS_FILE.read_text(encoding="utf-8"))
        presets = [Preset(**{k: v for k, v in item.items() if k in Preset.__dataclass_fields__}) for item in data]
    except (OSError, ValueError, TypeError):
        presets = []
    return presets or [Preset(name="Без пресета")]
