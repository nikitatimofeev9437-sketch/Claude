"""Paths and persistent settings."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
CHECKPOINTS_DIR = MODELS / "checkpoints"
LORAS_DIR = MODELS / "loras"
UPSCALERS_DIR = MODELS / "upscalers"
DETECTORS_DIR = MODELS / "detectors"
POSES_DIR = MODELS / "poses"
PRESETS_FILE = ROOT / "presets" / "styles.json"
OUTPUTS_DIR = ROOT / "outputs"
SETTINGS_FILE = ROOT / "settings.json"

CHECKPOINT_EXT = {".safetensors", ".ckpt"}
LORA_EXT = {".safetensors"}
UPSCALER_EXT = {".pth", ".safetensors", ".pt"}
POSE_EXT = {".png", ".jpg", ".jpeg", ".webp"}

# Small helper models fetched from Hugging Face on first use.
VAE_FP16_FIX_REPO = "madebyollin/sdxl-vae-fp16-fix"
POSE_ADAPTER_REPO = "TencentARC/t2i-adapter-openpose-sdxl-1.0"
DETECTOR_REPO = "Bingsu/adetailer"
DETECTOR_FILES = {"face": "face_yolov8n.pt", "hand": "hand_yolov8n.pt"}


@dataclass
class Settings:
    # "balanced": model CPU offload (fits 8 GB), "fast": keep everything in VRAM.
    vram_mode: str = "balanced"
    vae_fp16_fix: bool = True
    last_checkpoint: str = ""
    ui: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self) -> None:
        SETTINGS_FILE.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")


def list_files(folder: Path, exts: set[str]) -> list[str]:
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts)


def list_checkpoints() -> list[str]:
    """Single-file checkpoints plus diffusers-format folders (with model_index.json)."""
    items = list_files(CHECKPOINTS_DIR, CHECKPOINT_EXT)
    if CHECKPOINTS_DIR.is_dir():
        items += sorted(p.name for p in CHECKPOINTS_DIR.iterdir() if (p / "model_index.json").is_file())
    return items
