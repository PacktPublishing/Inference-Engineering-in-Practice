"""Thin, on-demand Modal wrapper for the Chapter 1 CUDA experiment.

Defining this app does not start remote compute. Run it only after the author
has checked current spend, estimated cost, and explicitly approved the command.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import modal

REMOTE_CODE_DIR = "/root/chapter1"
if Path(REMOTE_CODE_DIR).is_dir():
    sys.path.insert(0, REMOTE_CODE_DIR)

from model_config import (  # noqa: E402
    HF_CACHE_PATH,
    MODAL_GPU,
    MODAL_TIMEOUT_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    PYTHON_VERSION,
    REMOTE_PACKAGES,
)

CODE_DIR = Path(REMOTE_CODE_DIR) if Path(REMOTE_CODE_DIR).is_dir() else Path(__file__).parent
CHAPTER_DIR = CODE_DIR.parent
DEFAULT_ARTIFACT_DIR = CHAPTER_DIR / "artifacts"

app = modal.App("llm-inference-engineering-chapter-01")
cache_volume = modal.Volume.from_name(
    "llm-inference-engineering-hf-cache",
    create_if_missing=True,
)
image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .uv_pip_install(*REMOTE_PACKAGES)
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_HUB_CACHE": f"{HF_CACHE_PATH}/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONPATH": REMOTE_CODE_DIR,
        }
    )
    .add_local_dir(CODE_DIR, remote_path=REMOTE_CODE_DIR)
)


@app.function(
    image=image,
    gpu=MODAL_GPU,
    volumes={HF_CACHE_PATH: cache_volume},
    timeout=MODAL_TIMEOUT_SECONDS,
    min_containers=0,
    max_containers=1,
    scaledown_window=2,
)
def run_on_gpu() -> dict[str, Any]:
    """Load the model once and return all small experiment artifacts."""
    sys.path.insert(0, REMOTE_CODE_DIR)

    import torch
    import transformers
    from experiments import run_full_experiment
    from prompt_to_logits import load_model

    device = torch.device("cuda")
    tokenizer, model = load_model(
        device=device,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
    )
    properties = torch.cuda.get_device_properties(device)
    driver_version = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    environment = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver_version,
        "gpu": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_memory_bytes": properties.total_memory,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": str(next(model.parameters()).dtype),
        "seed": 7,
    }
    torch.manual_seed(environment["seed"])
    return run_full_experiment(tokenizer, model, environment)


@app.local_entrypoint()
def main(output_dir: str = str(DEFAULT_ARTIFACT_DIR)) -> None:
    """Run the approved remote function and write its returned data locally."""
    from experiments import write_experiment_artifacts
    from kv_cache import write_artifacts as write_cache_artifacts

    result = run_on_gpu.remote()
    destination = Path(output_dir)
    write_experiment_artifacts(result, destination)
    write_cache_artifacts(destination)
    print(f"Wrote Chapter 1 artifacts to {destination}")
