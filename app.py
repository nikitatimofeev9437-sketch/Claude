"""SemiReal Studio — local SDXL image generator. Start with run.bat (Windows) or `python app.py`."""
import os
import sys

# Keep Hugging Face downloads (configs, VAE fix, pose adapter, detectors) next to the program.
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hf_cache"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from studio.ui.main_window import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
