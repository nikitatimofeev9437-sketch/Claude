"""Image helpers: upscaling, outpaint canvas, masks, pose prep, face/hand detection."""
from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from . import config


def round8(v: float) -> int:
    return max(8, int(round(v / 8)) * 8)


def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- upscaling

class Upscaler:
    """ESRGAN-family upscalers via spandrel (4x-UltraSharp, RealESRGAN, ...), or plain Lanczos."""

    LANCZOS = "Lanczos (без модели)"

    def __init__(self, device: str, dtype: torch.dtype):
        self.device, self.dtype = device, dtype
        self._name: str | None = None
        self._model = None

    @classmethod
    def available(cls) -> list[str]:
        return [cls.LANCZOS] + config.list_files(config.UPSCALERS_DIR, config.UPSCALER_EXT)

    def upscale(self, img: Image.Image, width: int, height: int, name: str) -> Image.Image:
        if not name or name == self.LANCZOS or not (config.UPSCALERS_DIR / name).is_file():
            return img.resize((width, height), Image.LANCZOS)
        model = self._load(name)
        try:
            out = self._run_tiled(model, img.convert("RGB"))
        finally:
            model.to("cpu")
            free_vram()
        return out.resize((width, height), Image.LANCZOS)

    def _load(self, name: str):
        if self._name != name:
            from spandrel import ModelLoader

            self._model = ModelLoader().load_from_file(str(config.UPSCALERS_DIR / name)).eval()
            self._name = name
        model = self._model.to(self.device)
        if self.dtype == torch.float16:
            try:
                model.half()
            except Exception:  # some architectures do not support fp16
                model.float()
        return model

    @torch.no_grad()
    def _run_tiled(self, model, img: Image.Image, tile: int = 512, pad: int = 16) -> Image.Image:
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255).unsqueeze(0)
        _, _, h, w = x.shape
        s = model.scale
        out = torch.zeros(1, 3, h * s, w * s)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
                py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
                patch = x[:, :, py0:py1, px0:px1].to(self.device, model.dtype)
                r = model(patch).float().cpu()
                oy, ox = (y0 - py0) * s, (x0 - px0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = r[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
        arr = out.clamp(0, 1).mul(255).round().byte()[0].permute(1, 2, 0).numpy()
        return Image.fromarray(arr)


# ---------------------------------------------------------------- outpaint / masks

def outpaint_canvas(img: Image.Image, left: int, right: int, top: int, bottom: int,
                    overlap: int = 32, feather: int = 24) -> tuple[Image.Image, Image.Image]:
    """Extend ``img``; return (canvas pre-filled with smeared edges, mask of the area to paint)."""
    img = img.convert("RGB")
    w, h = img.size
    new_w, new_h = round8(w + left + right), round8(h + top + bottom)
    right += new_w - (w + left + right)
    bottom += new_h - (h + top + bottom)

    import cv2

    arr = np.asarray(img)
    ext = cv2.copyMakeBorder(arr, max(top, 0), max(bottom, 0), max(left, 0), max(right, 0), cv2.BORDER_REPLICATE)
    blurred = cv2.GaussianBlur(ext, (0, 0), sigmaX=24)
    rng = np.random.default_rng(0)
    noisy = np.clip(blurred.astype(np.int16) + rng.integers(-24, 25, blurred.shape), 0, 255).astype(np.uint8)
    canvas = Image.fromarray(noisy)
    canvas.paste(img, (left, top))

    mask = Image.new("L", (new_w, new_h), 255)
    keep = (
        left + (overlap if left else 0),
        top + (overlap if top else 0),
        left + w - (overlap if right else 0),
        top + h - (overlap if bottom else 0),
    )
    ImageDraw.Draw(mask).rectangle(keep, fill=0)
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
        # Never leave the new area partially painted.
        m = np.asarray(mask).copy()
        m[:top, :] = 255
        m[new_h - bottom:, :] = 255
        m[:, :left] = 255
        m[:, new_w - right:] = 255
        mask = Image.fromarray(m)
    return canvas, mask


def prepare_mask(mask: Image.Image, size: tuple[int, int], blur: int) -> Image.Image:
    mask = mask.convert("L").resize(size, Image.NEAREST)
    return mask.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else mask


def composite(generated: Image.Image, original: Image.Image, mask: Image.Image) -> Image.Image:
    """Keep the original pixels where the mask is black."""
    generated = generated.convert("RGB")
    original = original.convert("RGB").resize(generated.size)
    return Image.composite(generated, original, mask.convert("L").resize(generated.size))


def letterbox(img: Image.Image, width: int, height: int) -> Image.Image:
    """Fit an image into width x height on black, keeping proportions (for pose maps)."""
    img = img.convert("RGB")
    scale = min(width / img.width, height / img.height)
    resized = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


# ---------------------------------------------------------------- pose

_openpose = None


def extract_pose(img: Image.Image) -> Image.Image:
    """Turn a reference photo into an OpenPose skeleton image."""
    global _openpose
    try:
        from controlnet_aux.open_pose import OpenposeDetector
    except ImportError as e:
        raise RuntimeError("Для извлечения позы нужен пакет controlnet_aux (pip install controlnet_aux).") from e
    if _openpose is None:
        _openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    img = img.convert("RGB")
    res = max(img.size)
    return _openpose(img, detect_resolution=min(1024, res), image_resolution=res,
                     include_hand=True, include_face=True)


# ---------------------------------------------------------------- detection

class Detector:
    """YOLO face/hand detectors (the same models ADetailer uses), run on the CPU."""

    def __init__(self):
        self._models: dict[str, object] = {}

    def _model(self, kind: str):
        if kind not in self._models:
            try:
                from ultralytics import YOLO
            except ImportError as e:
                raise RuntimeError("Для исправления лиц/рук нужен пакет ultralytics.") from e
            path = config.DETECTORS_DIR / config.DETECTOR_FILES[kind]
            if not path.is_file():
                from huggingface_hub import hf_hub_download

                hf_hub_download(config.DETECTOR_REPO, config.DETECTOR_FILES[kind], local_dir=config.DETECTORS_DIR)
            self._models[kind] = YOLO(str(path))
        return self._models[kind]

    def detect(self, img: Image.Image, kind: str, confidence: float = 0.3) -> list[tuple[int, int, int, int]]:
        result = self._model(kind).predict(img.convert("RGB"), conf=confidence, device="cpu", verbose=False)[0]
        boxes = [tuple(int(v) for v in b) for b in result.boxes.xyxy.tolist()]
        # Largest first: the main subject gets fixed before background figures.
        return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def box_mask(size: tuple[int, int], box, dilate: float = 0.1, blur: int = 6) -> Image.Image:
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * dilate, (y1 - y0) * dilate
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(
        (max(0, x0 - dx), max(0, y0 - dy), min(size[0], x1 + dx), min(size[1], y1 + dy)), fill=255
    )
    return mask.filter(ImageFilter.GaussianBlur(blur)) if blur else mask


def save_png(img: Image.Image, folder: Path, seed: int, parameters: str) -> Path:
    from PIL.PngImagePlugin import PngInfo

    folder.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name[:5]) for p in folder.glob("*.png") if p.name[:5].isdigit()]
    path = folder / f"{(max(existing) + 1 if existing else 0):05d}-{seed}.png"
    info = PngInfo()
    info.add_text("parameters", parameters)
    img.save(path, pnginfo=info)
    return path
