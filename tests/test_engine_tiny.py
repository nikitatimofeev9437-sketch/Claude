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
