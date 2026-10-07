"""Generation engine: SDXL pipelines on diffusers, tuned for an 8 GB GPU."""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from . import config, imaging, prompting, safety

SAMPLERS: dict[str, tuple[str, dict]] = {
    "DPM++ 2M Karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "DPM++ 2M SDE Karras": ("DPMSolverMultistepScheduler",
                            {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"}),
    "DPM++ SDE Karras": ("DPMSolverSinglestepScheduler", {"use_karras_sigmas": True}),
    "Euler a": ("EulerAncestralDiscreteScheduler", {}),
    "Euler": ("EulerDiscreteScheduler", {}),
    "UniPC": ("UniPCMultistepScheduler", {}),
}

# Common SDXL buckets (~1 MP), portrait first.
RESOLUTIONS = [
    (832, 1216), (896, 1152), (768, 1344), (1024, 1024),
    (1216, 832), (1152, 896), (1344, 768),
]

MODES = ("txt2img", "img2img", "inpaint", "outpaint")


class Cancelled(Exception):
    pass


@dataclass
class LoraSpec:
    file: str
    weight: float = 0.8


@dataclass
class Job:
    mode: str = "txt2img"
    prompt: str = ""
    negative: str = ""
    width: int = 832
    height: int = 1216
    steps: int = 28
    cfg: float = 6.0
    sampler: str = "DPM++ 2M Karras"
    clip_skip: int = 2
    seed: int = -1
    batch: int = 1
    loras: list[LoraSpec] = field(default_factory=list)
    # img2img / inpaint / outpaint
    init_image: Image.Image | None = None
    mask: Image.Image | None = None
    strength: float = 0.6
    mask_blur: int = 8
    inpaint_only_masked: bool = False
    outpaint: tuple[int, int, int, int] = (0, 0, 0, 0)  # left, right, top, bottom
    # pose (txt2img only)
    pose_image: Image.Image | None = None
    pose_strength: float = 0.9
    # hires fix (txt2img / img2img)
    hires: bool = False
    hires_scale: float = 1.5
    hires_denoise: float = 0.35
    hires_steps: int = 15
    upscaler: str = imaging.Upscaler.LANCZOS
    # detailer
    fix_faces: bool = False
    fix_hands: bool = False
    detail_strength: float = 0.4
    detail_prompt: str = ""


ProgressFn = Callable[[str, int, int], None]


class Engine:
    def __init__(self, settings: config.Settings):
        self.settings = settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.base = None
        self.checkpoint: str | None = None
        self._derived: dict[str, object] = {}
        self._sched_config = None
        self._adapter = None
        self._loras: dict[str, float] = {}  # adapter name -> weight currently applied
        self.upscaler = imaging.Upscaler(self.device, self.dtype)
        self.detector = imaging.Detector()

    # ------------------------------------------------------------ loading

    def unload(self) -> None:
        self.base = None
        self.checkpoint = None
        self._derived.clear()
        self._adapter = None
        self._loras.clear()
        imaging.free_vram()

    def load(self, checkpoint: str, progress: ProgressFn | None = None) -> None:
        if self.base is not None and self.checkpoint == checkpoint:
            return
        from diffusers import AutoencoderKL, StableDiffusionXLPipeline

        self.unload()
        path = config.CHECKPOINTS_DIR / checkpoint
        if not path.exists():
            raise FileNotFoundError(f"Модель не найдена: {path}")
        if progress:
            progress(f"Загрузка модели {checkpoint}…", 0, 0)
        if path.is_dir():
            pipe = StableDiffusionXLPipeline.from_pretrained(str(path), torch_dtype=self.dtype)
        else:
            pipe = StableDiffusionXLPipeline.from_single_file(str(path), torch_dtype=self.dtype)
        if self.settings.vae_fp16_fix and self.dtype == torch.float16:
            # Most SDXL VAEs produce black/NaN images in fp16; this one does not.
            pipe.vae = AutoencoderKL.from_pretrained(config.VAE_FP16_FIX_REPO, torch_dtype=self.dtype)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.enable_slicing()
        self._sched_config = pipe.scheduler.config
        self.base = pipe
        self.checkpoint = checkpoint
        if self.device == "cuda" and self.settings.vram_mode == "fast":
            pipe.to("cuda")

    def _pose_adapter(self):
        if self._adapter is None:
            from diffusers import T2IAdapter

            self._adapter = T2IAdapter.from_pretrained(config.POSE_ADAPTER_REPO, torch_dtype=self.dtype)
            if self.device == "cuda" and self.settings.vram_mode == "fast":
                self._adapter.to("cuda")
        return self._adapter

    def _pipe(self, kind: str):
        """Return the pipeline for ``kind``; all of them share the base model's weights."""
        if self.base is None:
            raise RuntimeError("Модель не загружена.")
        if kind == "txt2img":
            pipe = self.base
        else:
            if kind not in self._derived:
                import diffusers

                cls, extra = {
                    "img2img": ("StableDiffusionXLImg2ImgPipeline", {}),
                    "inpaint": ("StableDiffusionXLInpaintPipeline", {}),
                    "pose": ("StableDiffusionXLAdapterPipeline", {"adapter": self._pose_adapter()}),
                }[kind]
                p = getattr(diffusers, cls).from_pipe(self.base, **extra)
                p.set_progress_bar_config(disable=True)
                self._derived[kind] = p
            pipe = self._derived[kind]
        if self.device == "cuda" and self.settings.vram_mode != "fast":
            # Re-attach offload hooks to whichever pipeline runs next (they share modules).
            pipe.enable_model_cpu_offload()
        return pipe

    def _set_sampler(self, pipe, name: str) -> None:
        import diffusers

        cls_name, kwargs = SAMPLERS.get(name, SAMPLERS["DPM++ 2M Karras"])
        pipe.scheduler = getattr(diffusers, cls_name).from_config(self._sched_config, **kwargs)

    def _apply_loras(self, loras: list[LoraSpec]) -> None:
        wanted: dict[str, tuple[str, float]] = {}
        for spec in loras:
            safety.check(Path(spec.file).stem)
            name = re.sub(r"\W", "_", Path(spec.file).stem)
            wanted[name] = (spec.file, spec.weight)
        if {k: w for k, (_, w) in wanted.items()} == self._loras:
            return
        self.base.remove_all_hooks()
        stale = [n for n in self._loras if n not in wanted]
        if stale and not wanted:
            self.base.unload_lora_weights()
        elif stale:
            self.base.delete_adapters(stale)
        for name, (file, _) in wanted.items():
            if name not in self._loras:
                path = config.LORAS_DIR / file
                if not path.is_file():
                    raise FileNotFoundError(f"LoRA не найдена: {path}")
                self.base.load_lora_weights(str(path), adapter_name=name)
        if wanted:
            self.base.set_adapters(list(wanted), adapter_weights=[w for _, w in wanted.values()])
        self._loras = {k: w for k, (_, w) in wanted.items()}

    # ------------------------------------------------------------ generation

    def generate(self, job: Job, progress: ProgressFn, cancelled: Callable[[], bool],
                 on_image: Callable[[Image.Image, Path, str], None] | None = None) -> list[Path]:
        if self.base is None:
            raise RuntimeError("Сначала выберите и загрузите модель.")
        if job.mode not in MODES:
            raise ValueError(f"Неизвестный режим: {job.mode}")
        safety.check(job.prompt, job.detail_prompt)
        negative = safety.harden_negative(job.negative)
        self._apply_loras(job.loras)

        seed = job.seed if job.seed >= 0 else random.randint(0, 2**32 - 1)
        saved: list[Path] = []
        folder = config.OUTPUTS_DIR / datetime.now().strftime("%Y-%m-%d")
        for i in range(max(1, job.batch)):
            if cancelled():
                raise Cancelled()
            s = seed + i
            label = f"[{i + 1}/{job.batch}] " if job.batch > 1 else ""
            img = self._generate_one(job, negative, s, lambda st, a, b: progress(label + st, a, b), cancelled)
            params = self._describe(job, negative, s)
            path = imaging.save_png(img, folder, s, params)
            saved.append(path)
            if on_image:
                on_image(img, path, params)
        imaging.free_vram()
        return saved

    def _callbacks(self, stage: str, progress: ProgressFn, cancelled, total_hint: int):
        def on_step_end(pipe, step, timestep, kwargs):
            if cancelled():
                raise Cancelled()
            progress(stage, step + 1, getattr(pipe, "num_timesteps", total_hint) or total_hint)
            return kwargs

        def legacy(step, timestep, latents):
            if cancelled():
                raise Cancelled()
            progress(stage, step + 1, total_hint)

        return on_step_end, legacy

    def _embeds(self, pipe, prompt: str, negative: str, clip_skip: int) -> dict:
        pe, ne, pp, np_ = prompting.encode_prompts(pipe, prompt, negative, clip_skip)
        return dict(prompt_embeds=pe, negative_prompt_embeds=ne,
                    pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=np_)

    def _vae_tiling(self, pipe, w: int, h: int) -> None:
        if w * h > 1536 * 1024:
            pipe.vae.enable_tiling()
        else:
            pipe.vae.disable_tiling()

    def _generate_one(self, job: Job, negative: str, seed: int, progress: ProgressFn, cancelled) -> Image.Image:
        gen = lambda: torch.Generator("cpu").manual_seed(seed)  # noqa: E731
        common = dict(guidance_scale=job.cfg)

        if job.mode == "txt2img":
            kind = "pose" if job.pose_image is not None else "txt2img"
            pipe = self._pipe(kind)
            self._set_sampler(pipe, job.sampler)
            emb = self._embeds(pipe, job.prompt, negative, job.clip_skip)
            w, h = imaging.round8(job.width), imaging.round8(job.height)
            self._vae_tiling(pipe, w, h)
            on_end, legacy = self._callbacks("Генерация", progress, cancelled, job.steps)
            if kind == "pose":
                img = pipe(**emb, **common, image=imaging.letterbox(job.pose_image, w, h), width=w, height=h,
                           num_inference_steps=job.steps, generator=gen(), callback=legacy,
                           adapter_conditioning_scale=job.pose_strength).images[0]
            else:
                img = pipe(**emb, **common, width=w, height=h, num_inference_steps=job.steps,
                           generator=gen(), callback_on_step_end=on_end).images[0]

        elif job.mode == "img2img":
            init = self._require_init(job)
            w, h = self._fit(init.width, init.height)
            pipe = self._pipe("img2img")
            self._set_sampler(pipe, job.sampler)
            emb = self._embeds(pipe, job.prompt, negative, job.clip_skip)
            self._vae_tiling(pipe, w, h)
            on_end, _ = self._callbacks("Img2img", progress, cancelled, job.steps)
            img = pipe(**emb, **common, image=init.convert("RGB").resize((w, h), Image.LANCZOS),
                       strength=job.strength, num_inference_steps=job.steps,
                       generator=gen(), callback_on_step_end=on_end).images[0]

        elif job.mode == "inpaint":
            init = self._require_init(job)
            if job.mask is None or not job.mask.getbbox():
                raise ValueError("Нарисуйте маску — область, которую нужно перерисовать.")
            w, h = self._fit(init.width, init.height)
            image = init.convert("RGB").resize((w, h), Image.LANCZOS)
            mask = imaging.prepare_mask(job.mask, (w, h), job.mask_blur)
            img = self._inpaint(image, mask, job, negative, job.prompt, job.strength, gen, progress, cancelled,
                                crop=job.inpaint_only_masked, stage="Inpaint")

        else:  # outpaint
            init = self._require_init(job)
            left, right, top, bottom = job.outpaint
            if not any((left, right, top, bottom)):
                raise ValueError("Укажите, на сколько пикселей расширить изображение.")
            canvas, mask = imaging.outpaint_canvas(init, left, right, top, bottom)
            if max(canvas.size) > 2048:
                raise ValueError("Итоговый размер больше 2048 px по стороне — расширяйте в несколько шагов.")
            img = self._inpaint(canvas, mask, job, negative, job.prompt, job.strength, gen, progress, cancelled,
                                crop=False, stage="Дорисовка")

        if job.hires and job.mode in ("txt2img", "img2img"):
            img = self._hires(img, job, negative, seed, progress, cancelled)
        for kind, enabled in (("face", job.fix_faces), ("hand", job.fix_hands)):
            if enabled:
                img = self._detail(img, kind, job, negative, seed, progress, cancelled)
        return img

    def _require_init(self, job: Job) -> Image.Image:
        if job.init_image is None:
            raise ValueError("Загрузите исходное изображение.")
        return job.init_image

    @staticmethod
    def _fit(w: int, h: int, max_side: int = 1536) -> tuple[int, int]:
        """Keep the source proportions but stay in a size SDXL handles on 8 GB."""
        scale = min(1.0, max_side / max(w, h))
        if w * h * scale * scale < 768 * 768:  # tiny inputs: upscale towards SDXL's native size
            scale = math.sqrt(1024 * 1024 / (w * h))
        return imaging.round8(w * scale), imaging.round8(h * scale)

    def _inpaint(self, image, mask, job, negative, prompt, strength, gen, progress, cancelled, crop, stage,
                 steps=None):
        pipe = self._pipe("inpaint")
        self._set_sampler(pipe, job.sampler)
        emb = self._embeds(pipe, prompt, negative, job.clip_skip)
        steps = steps or job.steps
        on_end, _ = self._callbacks(stage, progress, cancelled, steps)
        if crop:
            # Repaint only the masked region at full SDXL resolution, then paste it back.
            out = pipe(**emb, guidance_scale=job.cfg, image=image, mask_image=mask, strength=strength,
                       padding_mask_crop=32, width=1024, height=1024, num_inference_steps=steps,
                       generator=gen(), callback_on_step_end=on_end).images[0]
            return out
        self._vae_tiling(pipe, *image.size)
        out = pipe(**emb, guidance_scale=job.cfg, image=image, mask_image=mask, strength=strength,
                   width=image.width, height=image.height, num_inference_steps=steps,
                   generator=gen(), callback_on_step_end=on_end).images[0]
        return imaging.composite(out, image, mask)

    def _hires(self, img, job, negative, seed, progress, cancelled):
        w, h = imaging.round8(img.width * job.hires_scale), imaging.round8(img.height * job.hires_scale)
        progress("Апскейл", 0, 0)
        up = self.upscaler.upscale(img, w, h, job.upscaler)
        if cancelled():
            raise Cancelled()
        pipe = self._pipe("img2img")
        self._set_sampler(pipe, job.sampler)
        emb = self._embeds(pipe, job.prompt, negative, job.clip_skip)
        self._vae_tiling(pipe, w, h)
        steps = math.ceil(job.hires_steps / max(job.hires_denoise, 0.05))
        on_end, _ = self._callbacks("Hires fix", progress, cancelled, job.hires_steps)
        return pipe(**emb, guidance_scale=job.cfg, image=up, strength=job.hires_denoise,
                    num_inference_steps=steps, generator=torch.Generator("cpu").manual_seed(seed),
                    callback_on_step_end=on_end).images[0]

    def _detail(self, img, kind, job, negative, seed, progress, cancelled):
        name = "лица" if kind == "face" else "руки"
        progress(f"Поиск: {name}", 0, 0)
        boxes = self.detector.detect(img, kind)[:4]
        prompt = job.detail_prompt.strip() or job.prompt
        for i, box in enumerate(boxes):
            if cancelled():
                raise Cancelled()
            mask = imaging.box_mask(img.size, box)
            gen = lambda s=seed + 1000 + i: torch.Generator("cpu").manual_seed(s)  # noqa: E731
            steps = max(job.steps, math.ceil(12 / max(job.detail_strength, 0.05)))
            img = self._inpaint(img, mask, job, negative, prompt, job.detail_strength, gen, progress, cancelled,
                                crop=True, stage=f"Исправление {name} {i + 1}/{len(boxes)}", steps=steps)
        return img

    def _describe(self, job: Job, negative: str, seed: int) -> str:
        parts = [
            f"Steps: {job.steps}", f"Sampler: {job.sampler}", f"CFG scale: {job.cfg}", f"Seed: {seed}",
            f"Size: {job.width}x{job.height}", f"Model: {self.checkpoint}", f"Clip skip: {job.clip_skip}",
            f"Mode: {job.mode}",
        ]
        if job.mode != "txt2img":
            parts.append(f"Denoising strength: {job.strength}")
        if job.loras:
            # Quoted, so the comma-separated list survives parsing (see pnginfo.py).
            parts.append('LoRA: "' + ", ".join(f"{x.file}:{x.weight}" for x in job.loras) + '"')
        if job.pose_image is not None and job.mode == "txt2img":
            parts.append(f"Pose strength: {job.pose_strength}")
        if job.hires and job.mode in ("txt2img", "img2img"):
            parts.append(f"Hires: x{job.hires_scale} {job.upscaler} denoise {job.hires_denoise}")
        if job.fix_faces or job.fix_hands:
            parts.append(f"Detailer: faces={job.fix_faces} hands={job.fix_hands} strength={job.detail_strength}")
        return f"{job.prompt}\nNegative prompt: {negative}\n" + ", ".join(parts)
