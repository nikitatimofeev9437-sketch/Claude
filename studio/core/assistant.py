"""Prompt assistant backed by Grok.

Two ways to reach it:
* chat (works with a SuperGrok / X Premium subscription): the app builds a request,
  the user pastes it into grok.com and copies the answer back;
* xAI API (separately billed key from console.x.ai): the same request sent over HTTP.
Either way the answer is the same JSON, parsed by :func:`parse_answer`.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from . import safety
from .engine import RESOLUTIONS, SAMPLERS

GROK_CHAT_URL = "https://grok.com/"
XAI_API_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4"

_SYSTEM = """You write prompts for a local Stable Diffusion XL image generator (semi-realistic style).
Rules:
- All characters are consenting adults (21+). Never depict minors or childlike features, never use real people.
- The prompt is English, comma-separated booru-style tags and short phrases, most important first,
  A1111 weights like (tag:1.2) allowed sparingly. 40-90 words.
- Do NOT add quality/style prefix tags; the app adds them from the preset: {preset_prefix}
- Negative prompt: only scene-specific extras (the preset already has a generic negative).
Reply with ONE JSON object in a ```json code block and nothing else."""

_SCHEMA_PROMPT = """{
  "prompt": "...",
  "negative": "..."
}"""

_SCHEMA_FULL = """{
  "prompt": "...",
  "negative": "...",
  "width": 832, "height": 1216,
  "steps": 28, "cfg": 6.0, "sampler": "DPM++ 2M Karras",
  "hires": true, "fix_faces": true, "fix_hands": false,
  "comment": "one short sentence in Russian on why these settings"
}"""


def build_request(idea: str, preset_name: str, preset_prefix: str, with_settings: bool,
                  current_prompt: str = "") -> tuple[str, str]:
    """Return (system, user) messages."""
    system = _SYSTEM.format(preset_prefix=preset_prefix.strip() or "(none)")
    lines = [f"Preset: {preset_name}.", f"Idea (may be in Russian): {idea.strip()}"]
    if current_prompt.strip():
        lines.append(f"Current prompt to improve or build upon: {current_prompt.strip()}")
    if with_settings:
        lines += [
            "Also choose generation settings. GPU: RTX 4070 Laptop 8 GB; one image must take under 5 minutes.",
            "Allowed sizes: " + ", ".join(f"{w}x{h}" for w, h in RESOLUTIONS) + ".",
            "Allowed samplers: " + ", ".join(SAMPLERS) + ". Steps 20-40, cfg 3-9.",
            "JSON format:", _SCHEMA_FULL,
        ]
    else:
        lines += ["JSON format:", _SCHEMA_PROMPT]
    return system, "\n".join(lines)


def chat_text(system: str, user: str) -> str:
    """Single message to paste into the Grok chat."""
    return f"{system}\n\n{user}"


def ask_api(system: str, user: str, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 90) -> str:
    body = json.dumps({
        "model": model or DEFAULT_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.8,
    }).encode()
    req = urllib.request.Request(XAI_API_URL, data=body, headers={
        "Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        if e.code in (401, 403) or "api key" in detail.lower():
            raise RuntimeError(f"xAI отклонил ключ ({e.code}). Проверьте API-ключ и баланс на console.x.ai.\n{detail}")
        raise RuntimeError(f"Ошибка xAI API {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Нет связи с api.x.ai: {e.reason}") from e
    return data["choices"][0]["message"]["content"]


def _extract_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [m.group(1)] if m else []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            data = json.loads(c)
            if isinstance(data, dict):
                return data
        except ValueError:
            continue
    raise ValueError("В ответе Grok не найден JSON. Скопируйте ответ целиком (вместе с блоком ```json).")


def _num(v, lo, hi, cast):
    try:
        return max(lo, min(hi, cast(v)))
    except (TypeError, ValueError):
        return None


def parse_answer(text: str) -> dict:
    """Validate Grok's answer; returns only known, clamped fields."""
    data = _extract_json(text)
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("Grok не вернул поле \"prompt\".")
    out: dict = {"prompt": prompt, "negative": str(data.get("negative") or "").strip()}
    safety.check(out["prompt"])

    if "width" in data and "height" in data:
        w, h = _num(data["width"], 512, 2048, int), _num(data["height"], 512, 2048, int)
        if w and h:
            # Snap to the closest supported SDXL bucket by aspect ratio.
            out["size"] = min(RESOLUTIONS, key=lambda r: abs(r[0] / r[1] - w / h))
    for key, lo, hi, cast in (("steps", 10, 60, int), ("cfg", 1.0, 15.0, float)):
        if key in data and (v := _num(data[key], lo, hi, cast)) is not None:
            out[key] = v
    if data.get("sampler") in SAMPLERS:
        out["sampler"] = data["sampler"]
    for key in ("hires", "fix_faces", "fix_hands"):
        if isinstance(data.get(key), bool):
            out[key] = data[key]
    if data.get("comment"):
        out["comment"] = str(data["comment"])[:300]
    return out
