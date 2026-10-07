"""Read generation parameters back from PNGs (this app's own and A1111/Forge format)."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

from . import config, safety
from .engine import SAMPLERS

_RE_PARAM = re.compile(r'\s*([\w][\w \-/+]*):\s*("(?:\\.|[^\\"])*"|[^,]*)(?:,|$)')
_RE_SIZE = re.compile(r"^(\d+)x(\d+)$")
_RE_LORA_TAG = re.compile(r"<lora:([^:>]+):?([-\d.]*)[^>]*>")


def read_parameters(path: str | Path) -> str | None:
    with Image.open(path) as img:
        text = img.info.get("parameters")
        if not text and hasattr(img, "getexif"):
            # JPEG/WebP from A1111 keep it in EXIF UserComment.
            raw = img.getexif().get_ifd(0x8769).get(0x9286)
            if isinstance(raw, bytes):
                text = raw[8:].decode("utf-16-be" if raw[:8] == b"UNICODE\0" else "utf-8", errors="ignore")
            elif isinstance(raw, str):
                text = raw
    return text.strip() if text else None


def _find_lora(name: str) -> str | None:
    """Match a LoRA name from metadata to a file in models/loras."""
    stem = Path(name.strip()).stem.lower()
    for f in config.list_files(config.LORAS_DIR, config.LORA_EXT):
        if Path(f).stem.lower() == stem:
            return f
    return None


def parse_parameters(text: str) -> dict:
    """Return a dict of UI field values plus ``missing_loras`` and ``model``."""
    lines = text.strip().splitlines()
    params_line = lines[-1] if lines and _RE_PARAM.match(lines[-1]) and "Steps:" in lines[-1] else ""
    body = lines[:-1] if params_line else lines
    prompt_lines, neg_lines, in_neg = [], [], False
    for line in body:
        if line.startswith("Negative prompt:"):
            in_neg = True
            line = line[len("Negative prompt:"):].strip()
        (neg_lines if in_neg else prompt_lines).append(line)
    prompt = "\n".join(prompt_lines).strip()
    negative = "\n".join(neg_lines).strip()

    # The app appends its age guard to every negative; it is re-added on generation.
    guard = ", " + safety.ALWAYS_NEGATIVE
    if negative.endswith(guard):
        negative = negative[: -len(guard)]
    elif negative == safety.ALWAYS_NEGATIVE:
        negative = ""

    kv = {}
    for m in _RE_PARAM.finditer(params_line):
        key, value = m.group(1).strip(), m.group(2).strip()
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        kv[key] = value

    out: dict = {"prompt": prompt, "negative": negative, "loras": [], "missing_loras": []}

    # A1111 writes LoRAs as <lora:name:weight> inside the prompt; this app uses its own list.
    for name, weight in _RE_LORA_TAG.findall(prompt):
        f = _find_lora(name)
        w = float(weight) if weight else 1.0
        (out["loras"].append([f, w]) if f else out["missing_loras"].append(name))
    out["prompt"] = re.sub(r"\s*,?\s*" + _RE_LORA_TAG.pattern, "", prompt).strip(" ,")
    if kv.get("LoRA"):
        for item in kv["LoRA"].split(", "):
            name, _, weight = item.rpartition(":")
            f = _find_lora(name or item)
            try:
                w = float(weight)
            except ValueError:
                w = 0.8
            (out["loras"].append([f, w]) if f else out["missing_loras"].append(name or item))

    def num(key, cast):
        try:
            return cast(kv[key])
        except (KeyError, ValueError):
            return None

    for key, field, cast in (("Steps", "steps", int), ("CFG scale", "cfg", float), ("Seed", "seed", int),
                             ("Clip skip", "clip_skip", int), ("Denoising strength", "strength", float)):
        if (v := num(key, cast)) is not None:
            out[field] = v
    if m := _RE_SIZE.match(kv.get("Size", "")):
        out["size"] = (int(m.group(1)), int(m.group(2)))

    sampler = kv.get("Sampler", "")
    if kv.get("Schedule type", "").lower() == "karras" and not sampler.endswith("Karras"):
        sampler += " Karras"
    if sampler in SAMPLERS:
        out["sampler"] = sampler
    if kv.get("Mode") in ("txt2img", "img2img", "inpaint", "outpaint"):
        out["mode"] = kv["Mode"]
    if "Model" in kv:
        out["model"] = kv["Model"]

    # Hires: this app writes "Hires: x1.5 <upscaler> denoise 0.35"; A1111 writes separate keys.
    if m := re.match(r"x([\d.]+) (.+) denoise ([\d.]+)$", kv.get("Hires", "")):
        out.update(hires=True, hires_scale=float(m.group(1)), upscaler=m.group(2),
                   hires_denoise=float(m.group(3)))
    elif "Hires upscale" in kv:
        out["hires"] = True
        if (v := num("Hires upscale", float)) is not None:
            out["hires_scale"] = v
        if (v := num("Denoising strength", float)) is not None:
            out["hires_denoise"] = v
            out.pop("strength", None)
        if (v := num("Hires steps", int)):
            out["hires_steps"] = v
    if m := re.search(r"faces=(\w+) hands=(\w+) strength=([\d.]+)", kv.get("Detailer", "")):
        out.update(fix_faces=m.group(1) == "True", fix_hands=m.group(2) == "True",
                   detail_strength=float(m.group(3)))
    elif "ADetailer model" in kv:
        out["fix_faces"] = "face" in kv["ADetailer model"]
    return out
