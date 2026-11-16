"""One temporary engine/API process and loopback client. Importing spends nothing.

Run only after the author reports usable budget for this exact attempt.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

REMOTE = Path("/root/chapter2")
LOCAL = Path(__file__).resolve().parents[1]
CHAPTER = REMOTE if REMOTE.exists() else LOCAL
sys.path.insert(0, str(CHAPTER / "code"))

IMAGE = "vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b"
MODEL_PATH = Path("/cache/qwen2.5-1.5b-989aa798")
BASE_URL = "http://127.0.0.1:8000/v1"
app = modal.App("inference-book-chapter-02")
cache = modal.Volume.from_name("inference-book-chapter-02-model", create_if_missing=True)
evidence = modal.Volume.from_name("inference-book-chapter-02-evidence", create_if_missing=True)
image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    # vLLM's Ubuntu image supplies python3; Modal Functions also require python on PATH.
    # Alias the existing interpreter to preserve its installed engine dependencies.
    .run_commands(
        "if ! command -v python >/dev/null 2>&1; then "
        "ln -s /usr/bin/python3 /usr/local/bin/python; fi",
        "python --version && python -m pip --version",
        "timeout 180 python3 -m pip install --no-deps --retries 0 --timeout 30 httpx==0.28.1",
    )
    .env({"HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1", "PYTHONPATH": str(REMOTE / "code")})
    .add_local_dir(LOCAL / "code", str(REMOTE / "code"))
    .add_local_file(LOCAL / "experiment.json", str(REMOTE / "experiment.json"))
)


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(8192, 8192),
    timeout=600,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    volumes={"/cache": cache},
)
def prepare_weights():
    completed = subprocess.run(
        [
            "python3",
            str(REMOTE / "code/prepare_model.py"),
            str(REMOTE / "experiment.json"),
            str(MODEL_PATH),
        ],
        timeout=540,
        check=False,
    )
    cache.commit()
    if completed.returncode:
        raise RuntimeError("CPU preparation failed; no GPU invocation will follow")
    return json.loads((MODEL_PATH / "preparation.json").read_text())


def server_command(config):
    return [
        "vllm",
        "serve",
        str(MODEL_PATH),
        "--served-model-name",
        config["model"],
        "--tokenizer",
        str(MODEL_PATH),
        "--dtype",
        "bfloat16",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--max-model-len",
        "1024",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.75",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--seed",
        "17",
        "--enforce-eager",
    ]


@app.function(
    image=image,
    gpu="L4",
    cpu=(2, 2),
    memory=(16384, 16384),
    timeout=1320,
    startup_timeout=300,
    retries=0,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    volumes={"/cache": cache, "/evidence": evidence},
)
async def measure(run_id: str, source: dict):
    import httpx
    from experiment import run_experiment
    from records import write_json
    from server_process import running_server

    directory = Path("/evidence") / run_id
    directory.mkdir(exist_ok=False)
    config = json.loads((REMOTE / "experiment.json").read_text())
    metadata = {
        "evidence_kind": "GPU measurement",
        "engine": "vLLM",
        "engine_image": IMAGE,
        "api_server": "vLLM packaged OpenAI-compatible server",
        "serving_framework": None,
        "platform": "temporary Modal Function",
        "topology": "client and API server in same container; loopback HTTP",
        "excluded": "Internet and laptop-to-cloud request latency",
        "resources": {"gpu": "L4:1", "cpu_cores": 2, "memory_mib": 16384},
        "source": source,
        "command": server_command(config),
    }
    outcome = {"status": "incomplete", "error": None}
    write_json(directory / "launch.json", metadata)
    await evidence.commit.aio()
    try:
        metadata["versions"] = {
            p: importlib.metadata.version(p) for p in ["vllm", "torch", "transformers", "httpx"]
        }
        if metadata["versions"]["vllm"] != "0.29.0":
            raise ValueError("Unexpected vLLM version in pinned image")
        metadata["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True, timeout=10)
        metadata["model_preparation"] = json.loads((MODEL_PATH / "preparation.json").read_text())
        async with asyncio.timeout(1260):
            async with running_server(
                server_command(config),
                BASE_URL,
                directory / "server.log",
                readiness_s=300,
                shutdown_s=10,
            ):
                async with httpx.AsyncClient(timeout=5, trust_env=False) as observer:
                    response = await observer.get("http://127.0.0.1:8000/metrics")
                    response.raise_for_status()
                    (directory / "metrics-before.txt").write_text(response.text)
                    cache_lines = [
                        line
                        for line in response.text.splitlines()
                        if line.startswith("vllm:cache_config_info{")
                    ]
                    if not cache_lines or not all(
                        'enable_prefix_caching="False"' in line for line in cache_lines
                    ):
                        raise ValueError("Cannot verify disabled prefix cache from engine metrics")
                    metadata["prefix_cache_verification"] = cache_lines
                    outcome = await run_experiment(BASE_URL, config, directory / "run", metadata)
                    response = await observer.get("http://127.0.0.1:8000/metrics")
                    (directory / "metrics-after.txt").write_text(response.text)
    except (Exception, asyncio.CancelledError) as exc:
        outcome = {"status": "incomplete", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        write_json(directory / "outcome.json", outcome)
        await evidence.commit.aio()
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@app.local_entrypoint()
def main(output: str):
    """The caller must check current budget before executing this entrypoint."""
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise ValueError("Choose a fresh output directory")
    config = json.loads((LOCAL / "experiment.json").read_text())
    from experiment import validate

    validate(config)
    source = {
        "files": {
            str(p.relative_to(LOCAL)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((LOCAL / "code").glob("*.py"))
        },
        "config_sha256": hashlib.sha256((LOCAL / "experiment.json").read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(
            ["git", "-C", str(LOCAL), "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "-C", str(LOCAL), "status", "--short"], text=True
        ),
    }
    run_id = datetime.now(UTC).strftime("vllm-%Y%m%dT%H%M%SZ")
    print(f"Run ID: {run_id}; partial evidence volume: inference-book-chapter-02-evidence")
    prepare_weights.remote()
    files = measure.remote(run_id, source)
    target.mkdir(parents=True)
    for name, contents in files.items():
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    outcome = json.loads((target / "outcome.json").read_text())
    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": outcome["status"],
                "error": outcome.get("error"),
                "output": str(target),
            },
            indent=2,
        )
    )
    if outcome["status"] != "completed":
        raise SystemExit(1)
