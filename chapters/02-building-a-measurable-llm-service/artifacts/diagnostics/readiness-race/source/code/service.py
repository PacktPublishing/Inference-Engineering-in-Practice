"""Run an engine service and its measurement client on an ordinary Linux GPU host.

This module has no Modal dependency. Use the engine image or a compatible
Python environment; provider provisioning and artifact transfer live elsewhere.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import subprocess
from pathlib import Path

import httpx
from engines import VERSIONS, engine_name, server_command
from experiment import run_experiment
from records import write_json
from server_process import running_server

BASE_URL = "http://127.0.0.1:8000/v1"


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
    outcome = {"status": "incomplete", "error": None}
    write_json(directory / "launch.json", metadata)
    try:
        metadata["versions"] = {
            p: importlib.metadata.version(p) for p in [name, "torch", "transformers", "httpx"]
        }
        if metadata["versions"][name] != VERSIONS[name]:
            raise ValueError("Unexpected engine version in pinned image")
        metadata["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, timeout=10)
        metadata["model_preparation"] = json.loads((model_path / "preparation.json").read_text())
        async with asyncio.timeout(1920):
            async with running_server(
                server_command(config, model_path),
                BASE_URL,
                directory / "server.log",
                readiness_s=900,
                shutdown_s=10,
            ):
                async with httpx.AsyncClient(timeout=5, trust_env=False) as observer:
                    response = await observer.get("http://127.0.0.1:8000/metrics")
                    response.raise_for_status()
                    (directory / "metrics-before.txt").write_text(response.text)
                    if name == "vllm":
                        cache_lines = [
                            line
                            for line in response.text.splitlines()
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
                    response = await observer.get("http://127.0.0.1:8000/metrics")
                    (directory / "metrics-after.txt").write_text(response.text)
    except (Exception, asyncio.CancelledError) as exc:
        outcome = {"status": "incomplete", "error": f"{type(exc).__name__}: {exc}"}
    finally:
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
