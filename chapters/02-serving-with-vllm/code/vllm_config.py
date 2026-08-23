"""Pinned, locally inspectable configuration for Chapter 2."""

from __future__ import annotations

from dataclasses import dataclass, field

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SERVED_MODEL_NAME = "chapter-2-qwen2.5-1.5b"

PYTHON_VERSION = "3.12"
VLLM_VERSION = "0.27.1"
CUDA_BASE_IMAGE = "nvidia/cuda:12.9.1-devel-ubuntu22.04"
REMOTE_PACKAGES = (f"vllm=={VLLM_VERSION}",)

MODAL_APP_NAME = "llm-inference-engineering-chapter-02"
MODAL_GPU = "L4"
MODAL_TIMEOUT_SECONDS = 30 * 60
MODAL_STARTUP_TIMEOUT_SECONDS = 20 * 60
MODAL_SCALEDOWN_WINDOW_SECONDS = 60
MODAL_MIN_CONTAINERS = 0
MODAL_MAX_CONTAINERS = 1

VLLM_PORT = 8000
HF_CACHE_PATH = "/cache/huggingface"
VLLM_CACHE_PATH = "/cache/vllm"

ARTIFACT_SCHEMA_VERSION = 1
WORKLOAD_SCHEMA_VERSION = 1
RANDOM_SEED = 17


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Arguments that materially affect a Chapter 2 vLLM server."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    served_model_name: str = SERVED_MODEL_NAME
    host: str = "0.0.0.0"
    port: int = VLLM_PORT
    dtype: str = "bfloat16"
    max_model_len: int = 4096
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = False
    enable_per_request_metrics: bool = True
    enable_mfu_metrics: bool = True
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if not self.model_revision:
            raise ValueError("model_revision cannot be empty")
        if not self.served_model_name:
            raise ValueError("served_model_name cannot be empty")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if any(not argument for argument in self.extra_args):
            raise ValueError("extra_args cannot contain empty arguments")


DEFAULT_SERVER_CONFIG = ServerConfig()
