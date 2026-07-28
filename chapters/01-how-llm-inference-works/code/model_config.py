"""Pinned configuration shared by the Chapter 1 examples."""

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

PYTHON_VERSION = "3.12"
TORCH_VERSION = "2.13.0"
TRANSFORMERS_VERSION = "5.14.1"
SAFETENSORS_VERSION = "0.8.0"

REMOTE_PACKAGES = (
    f"torch=={TORCH_VERSION}",
    f"transformers=={TRANSFORMERS_VERSION}",
    f"safetensors=={SAFETENSORS_VERSION}",
)

MODAL_GPU = "L4"
MODAL_TIMEOUT_SECONDS = 10 * 60
HF_CACHE_PATH = "/cache/huggingface"
