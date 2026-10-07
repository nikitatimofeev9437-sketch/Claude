"""Smoke test of the engine on a tiny random SDXL pipeline (CPU, downloads ~10 MB).

Run: python -m pytest tests/test_engine_tiny.py  (not needed for normal use)
"""
import shutil

import pytest
from PIL import Image

from studio.core import config, engine, prompting, safety


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    from huggingface_hub import snapshot_download

    src = snapshot_download("hf-internal-testing/tiny-stable-diffusion-xl-pipe")
    dst = config.CHECKPOINTS_DIR / "_tiny_test"
    shutil.copytree(src, dst, dirs_exist_ok=True)
    config.OUTPUTS_DIR = tmp_path_factory.mktemp("out")
    # Real inputs are resized towards ~1 MP; keep the tiny test images tiny.
    engine.Engine._fit = staticmethod(lambda w, h, max_side=1536: (w, h))
    e = engine.Engine(config.Settings(vae_fp16_fix=False))
    e.load("_tiny_test")
    yield e
    shutil.rmtree(dst, ignore_errors=True)


def run(e, **kw):
    job = engine.Job(width=64, height=64, steps=2, seed=1, **kw)
    return e.generate(job, lambda *a: None, lambda: False)


def test_parse_weights():
    segs = prompting.parse_prompt_attention("a (red:1.5) [cat] BREAK dog")
    assert ["red", 1.5] in segs
    assert any(s[1] == -1 for s in segs)


def test_safety():
    with pytest.raises(safety.BlockedPrompt):
        safety.check("a schoolgirl")
    with pytest.raises(safety.BlockedPrompt):
        safety.check("девочка в лесу")
    safety.check("adult woman, 25 years old, kidney beans")


def test_modes(eng):
    out = run(eng, prompt="woman, " * 60 + "(portrait:1.3)")  # >75 tokens
    img = Image.open(out[0])
    assert img.size == (64, 64)
    assert "parameters" in img.info
    run(eng, mode="img2img", init_image=img, strength=0.5)
    mask = Image.new("L", img.size, 0)
    mask.paste(255, (16, 16, 48, 48))
    run(eng, mode="inpaint", init_image=img, mask=mask, strength=0.8)
    out = run(eng, mode="outpaint", init_image=img, outpaint=(16, 16, 0, 8), strength=1.0)
    assert Image.open(out[0]).size == (96, 72)
    run(eng, prompt="x", hires=True, hires_scale=1.5, hires_steps=2, hires_denoise=0.5)


def test_cancel(eng):
    with pytest.raises(engine.Cancelled):
        eng.generate(engine.Job(width=64, height=64, steps=2), lambda *a: None, lambda: True)


def test_lora(eng):
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from diffusers import StableDiffusionXLPipeline

    unet = eng.base.unet
    unet.add_adapter(LoraConfig(r=4, target_modules=["to_q", "to_k", "to_v"]), adapter_name="tmp")
    sd = get_peft_model_state_dict(unet, adapter_name="tmp")
    unet.delete_adapters("tmp")
    path = config.LORAS_DIR / "_tiny_lora.safetensors"
    StableDiffusionXLPipeline.save_lora_weights(str(config.LORAS_DIR), unet_lora_layers=sd,
                                                weight_name=path.name)
    try:
        run(eng, prompt="x", loras=[engine.LoraSpec(path.name, 0.7)])
        run(eng, prompt="x", loras=[engine.LoraSpec(path.name, 0.3)])
        run(eng, prompt="x")
        assert eng._loras == {}
        with pytest.raises(safety.BlockedPrompt):
            run(eng, prompt="x", loras=[engine.LoraSpec("loli_style.safetensors", 1.0)])
    finally:
        path.unlink(missing_ok=True)


def test_grok_answer_parsing():
    from studio.core import assistant

    system, user = assistant.build_request("рыжая девушка на крыше", "Pony", "score_9, ", True)
    assert "score_9" in system and "832x1216" in user
    r = assistant.parse_answer('Вот:\n```json\n{"prompt": "adult woman, red hair", "negative": "blurry", '
                               '"width": 900, "height": 1300, "steps": 500, "cfg": 6, "sampler": "Euler a", '
                               '"hires": true, "fix_faces": "yes"}\n```')
    assert r["size"] == (832, 1216) and r["steps"] == 60 and r["sampler"] == "Euler a"
    assert r["hires"] is True and "fix_faces" not in r
    with pytest.raises(safety.BlockedPrompt):
        assistant.parse_answer('{"prompt": "schoolgirl"}')
    with pytest.raises(ValueError):
        assistant.parse_answer("no json here")


def test_pnginfo_roundtrip(eng, monkeypatch):
    from studio.core import pnginfo

    lora = config.LORAS_DIR / "_meta_lora.safetensors"
    lora.write_bytes(b"")
    try:
        job = engine.Job(width=64, height=64, steps=3, seed=42, prompt="adult woman, (red hair:1.2)",
                         negative="blurry", cfg=5.5, sampler="Euler a", clip_skip=2,
                         loras=[engine.LoraSpec("_meta_lora.safetensors", 0.6),
                                engine.LoraSpec("gone.safetensors", 0.4)],
                         hires=True, hires_scale=1.5, hires_denoise=0.3, fix_faces=True, detail_strength=0.45)
        p = pnginfo.parse_parameters(eng._describe(job, safety.harden_negative(job.negative), 42))
    finally:
        lora.unlink()
    assert p["prompt"] == job.prompt and p["negative"] == "blurry"
    assert (p["steps"], p["cfg"], p["seed"], p["sampler"], p["size"]) == (3, 5.5, 42, "Euler a", (64, 64))
    assert p["loras"] == [["_meta_lora.safetensors", 0.6]] and p["missing_loras"] == ["gone.safetensors"]
    assert p["hires"] and p["hires_scale"] == 1.5 and p["hires_denoise"] == 0.3
    assert p["fix_faces"] and not p["fix_hands"] and p["detail_strength"] == 0.45

    out = run(eng, prompt="adult woman")
    assert pnginfo.parse_parameters(pnginfo.read_parameters(out[0]))["seed"] == 1


def test_pnginfo_a1111():
    from studio.core import pnginfo

    text = ("masterpiece, woman <lora:someStyle:0.7>, smile\nNegative prompt: bad hands, \"quoted\"\n"
            "Steps: 30, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7, Seed: 123, Size: 832x1216, "
            "Model hash: abc, Model: myModel, Denoising strength: 0.4, Hires upscale: 2, Hires steps: 12, "
            "Hires upscaler: 4x-UltraSharp, ADetailer model: face_yolov8n.pt, Version: v1.10")
    p = pnginfo.parse_parameters(text)
    assert p["prompt"] == "masterpiece, woman, smile"
    assert p["missing_loras"] == ["someStyle"]
    assert p["sampler"] == "DPM++ 2M Karras" and p["size"] == (832, 1216) and p["model"] == "myModel"
    assert p["hires"] and p["hires_scale"] == 2.0 and p["hires_denoise"] == 0.4 and p["hires_steps"] == 12
    assert "strength" not in p and p["fix_faces"]


def test_library(tmp_path, monkeypatch):
    from studio.core import library

    monkeypatch.setattr(library, "LIBRARY_FILE", tmp_path / "prompts.json")
    library.save("a", {"prompt": "x"})
    library.save("b", {"prompt": "y"})
    entries = library.save("a", {"prompt": "z"})
    assert [e["name"] for e in entries] == ["a", "b"] and entries[0]["ui"]["prompt"] == "z"
    assert [e["name"] for e in library.delete("a")] == ["b"]
    assert [e["name"] for e in library.load()] == ["b"]
