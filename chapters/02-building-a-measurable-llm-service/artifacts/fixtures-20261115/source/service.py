"""Run an engine service and its measurement client on an ordinary Linux GPU host.

This module has no Modal dependency. Use the engine image or a compatible
Python environment; provider provisioning and artifact transfer live elsewhere.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
from pathlib import Path

import httpx
from engines import VERSIONS, engine_name, server_command
from experiment import run_experiment
from prepare_model import validate_preparation
from records import write_json
from server_process import running_server

BASE_URL = "http://127.0.0.1:8000/v1"


REQUIRED_METRICS = {
    "vllm": ("vllm:num_requests_running", "vllm:num_requests_waiting"),
    "sglang": ("sglang:num_running_reqs", "sglang:num_queue_reqs"),
}


def validate_metrics(text: str, engine: str) -> dict[str, list[float]]:
    """Validate the queue gauges used in the lesson, not every Prometheus field."""
    values = {}
    for name in REQUIRED_METRICS[engine]:
        pattern = re.compile(r"^" + re.escape(name) + r"(?:\{[^\n]*\})?\s+(\S+)(?:\s+\S+)?$")
        samples = [float(m.group(1)) for line in text.splitlines() if (m := pattern.match(line))]
        if not samples or any(not math.isfinite(v) or v < 0 for v in samples):
            raise ValueError(f"Missing or invalid required metric: {name}")
        values[name] = samples
    return values


async def capture_metrics(observer, directory: Path, engine: str, phase: str) -> str:
    """Save failed observations separately; an error page is never valid metrics."""
    observation = {"phase": phase, "status": "incomplete", "http_status": None}
    try:
        response = await observer.get("http://127.0.0.1:8000/metrics")
        observation["http_status"] = response.status_code
        response.raise_for_status()
        observation["required_gauges"] = validate_metrics(response.text, engine)
        (directory / f"metrics-{phase}.txt").write_text(response.text)
        observation["status"] = "completed"
        return response.text
    except Exception as exc:
        observation["error"] = f"{type(exc).__name__}: {exc}"
        if "response" in locals():
            (directory / f"metrics-{phase}-failed.txt").write_text(response.text)
        raise
    finally:
        write_json(directory / f"metrics-{phase}-observation.json", observation)


async def serve_and_measure(
    config: dict, model_path: Path, directory: Path, metadata: dict | None = None
) -> dict:
    name = engine_name(config)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(metadata or {})
    metadata.update(
        {
            "evidence_kind": "GPU measurement",
            "engine": name,
            "api_server": f"{name} packaged OpenAI-compatible server",
            "serving_framework": None,
            "topology": "client and API server on same host; loopback HTTP",
            "excluded": "Internet and laptop-to-cloud request latency",
            "command": server_command(config, model_path),
            "execution_mode": "eager"
            if config.get("engine", {}).get("enforce_eager")
            else "standard engine execution",
        }
    )
    environment = {"SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0"} if name == "sglang" else {}
    metadata["readiness"] = {
        "url": "http://127.0.0.1:8000/health",
        "environment": environment,
        "generates_requests": False,
    }
    source_dir = directory / "source"
    (source_dir / "code").mkdir(parents=True)
    for path in Path(__file__).parent.glob("*.py"):
        (source_dir / "code" / path.name).write_bytes(path.read_bytes())
    write_json(source_dir / "experiment.json", config)
    metadata["source"] = {
        **metadata.get("source", {}),
        "files": {
            str(p.relative_to(source_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source_dir / "code").glob("*.py"))
        },
        "config_sha256": hashlib.sha256((source_dir / "experiment.json").read_bytes()).hexdigest(),
    }
    outcome = {"status": "incomplete", "error": None}
    experiment_status = "not_started"
    write_json(directory / "launch.json", metadata)
    try:
        metadata["versions"] = {
            p: importlib.metadata.version(p) for p in [name, "torch", "transformers", "httpx"]
        }
        if metadata["versions"][name] != VERSIONS[name]:
            raise ValueError("Unexpected engine version in pinned image")
        metadata["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, timeout=10)
        metadata["model_preparation"] = json.loads((model_path / "preparation.json").read_text())
        validate_preparation(config, metadata["model_preparation"])
        write_json(directory / "launch.json", metadata)
        async with asyncio.timeout(1920):
            async with running_server(
                server_command(config, model_path),
                BASE_URL,
                directory / "server.log",
                readiness_s=900,
                shutdown_s=10,
                readiness_url="http://127.0.0.1:8000/health",
                environment=environment,
            ):
                async with httpx.AsyncClient(timeout=5, trust_env=False) as observer:
                    metrics = await capture_metrics(observer, directory, name, "before")
                    if name == "vllm":
                        cache_lines = [
                            line
                            for line in metrics.splitlines()
                            if line.startswith("vllm:cache_config_info{")
                        ]
                        if not cache_lines or not all(
                            'enable_prefix_caching="False"' in line for line in cache_lines
                        ):
                            raise ValueError("Cannot verify disabled vLLM prefix cache")
                        metadata["prefix_cache_verification"] = cache_lines
                    else:
                        response = await observer.get("http://127.0.0.1:8000/server_info")
                        response.raise_for_status()
                        info = response.json()
                        write_json(directory / "server-info.json", info)
                        if info.get("disable_radix_cache") is not True:
                            raise ValueError("Cannot verify disabled SGLang prefix cache")
                        metadata["prefix_cache_verification"] = {
                            "source": "/server_info",
                            "disable_radix_cache": True,
                        }
                    outcome = await run_experiment(BASE_URL, config, directory / "run", metadata)
                    experiment_status = outcome["status"]
                    await capture_metrics(observer, directory, name, "after")
    except (Exception, asyncio.CancelledError) as exc:
        outcome = {"status": "incomplete", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        outcome["experiment_status"] = experiment_status
        write_json(directory / "launch.json", metadata)
        write_json(directory / "outcome.json", outcome)
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--engine", choices=["vllm", "sglang"])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.engine:
        config["engine"]["name"] = args.engine
    result = asyncio.run(
        serve_and_measure(
            config,
            args.model_path,
            args.output,
            {"platform": "user-provided Linux GPU host or container"},
        )
    )
    print(json.dumps({"status": result["status"], "error": result.get("error")}, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
