"""Thin Modal wrapper and pure command builder for the Chapter 2 vLLM server.

Importing this module does not start remote compute. Any `modal serve` or
`modal deploy` command still requires the author's explicit approval.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping

import modal
from vllm_config import (
    CUDA_BASE_IMAGE,
    DEFAULT_SERVER_CONFIG,
    HF_CACHE_PATH,
    MODAL_APP_NAME,
    MODAL_GPU,
    MODAL_MAX_CONTAINERS,
    MODAL_MIN_CONTAINERS,
    MODAL_SCALEDOWN_WINDOW_SECONDS,
    MODAL_STARTUP_TIMEOUT_SECONDS,
    MODAL_TIMEOUT_SECONDS,
    PYTHON_VERSION,
    REMOTE_PACKAGES,
    VLLM_CACHE_PATH,
    VLLM_PORT,
    ServerConfig,
)


def build_serve_command(config: ServerConfig = DEFAULT_SERVER_CONFIG) -> tuple[str, ...]:
    """Return the exact `vllm serve` argument vector for one server."""
    command = [
        "vllm",
        "serve",
        config.model_id,
        "--revision",
        config.model_revision,
        "--served-model-name",
        config.served_model_name,
        "--host",
        config.host,
        "--port",
        str(config.port),
        "--dtype",
        config.dtype,
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--generation-config",
        "vllm",
    ]
    if config.enable_prefix_caching:
        command.append("--enable-prefix-caching")
    if config.enable_per_request_metrics:
        command.append("--enable-per-request-metrics")
    if config.enable_mfu_metrics:
        command.append("--enable-mfu-metrics")
    command.extend(config.extra_args)
    return tuple(command)


def server_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return explicit cache and telemetry settings for the server process."""
    environment = {
        "HF_HOME": HF_CACHE_PATH,
        "HF_HUB_CACHE": f"{HF_CACHE_PATH}/hub",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
    }
    if overrides is not None:
        environment.update(overrides)
    return environment


app = modal.App(MODAL_APP_NAME)
hf_cache_volume = modal.Volume.from_name(
    "llm-inference-engineering-hf-cache",
    create_if_missing=True,
)
vllm_cache_volume = modal.Volume.from_name(
    "llm-inference-engineering-vllm-cache",
    create_if_missing=True,
)
image = (
    modal.Image.from_registry(CUDA_BASE_IMAGE, add_python=PYTHON_VERSION)
    .entrypoint([])
    .uv_pip_install(*REMOTE_PACKAGES)
    .env(server_environment())
)


@app.function(
    image=image,
    gpu=MODAL_GPU,
    volumes={
        HF_CACHE_PATH: hf_cache_volume,
        VLLM_CACHE_PATH: vllm_cache_volume,
    },
    timeout=MODAL_TIMEOUT_SECONDS,
    min_containers=MODAL_MIN_CONTAINERS,
    max_containers=MODAL_MAX_CONTAINERS,
    scaledown_window=MODAL_SCALEDOWN_WINDOW_SECONDS,
)
@modal.concurrent(max_inputs=DEFAULT_SERVER_CONFIG.max_num_seqs)
@modal.web_server(
    port=VLLM_PORT,
    startup_timeout=MODAL_STARTUP_TIMEOUT_SECONDS,
)
def serve() -> None:
    """Start the pinned OpenAI-compatible server inside a Modal container."""
    environment = os.environ.copy()
    environment.update(server_environment())
    subprocess.Popen(build_serve_command(), env=environment)
