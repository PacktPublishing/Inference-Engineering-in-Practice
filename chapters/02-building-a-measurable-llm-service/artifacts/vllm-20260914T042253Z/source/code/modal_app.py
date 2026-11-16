"""One temporary engine/API process and loopback client. Importing spends nothing.

Modal provides compute and artifact transfer; the serving lesson lives in service.py.
"""

from __future__ import annotations

import hashlib
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
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=600,
    startup_timeout=600,
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


@app.function(
    image=image,
    gpu="L4",
    cpu=(4, 4),
    memory=(16384, 16384),
    timeout=2100,
    startup_timeout=600,
    retries=0,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    volumes={"/cache": cache, "/evidence": evidence},
)
async def measure(run_id: str, source: dict):
    from service import serve_and_measure

    directory = Path("/evidence") / run_id
    config = json.loads((REMOTE / "experiment.json").read_text())
    try:
        await serve_and_measure(
            config,
            MODEL_PATH,
            directory,
            {
                "engine_image": IMAGE,
                "platform": "temporary Modal Function",
                "resources": {"gpu": "L4:1", "cpu_cores": 4, "memory_mib": 16384},
                "source": source,
            },
        )
    finally:
        await evidence.commit.aio()
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@app.local_entrypoint()
def main(output: str):
    """Provision the declared experiment and return its evidence bundle."""
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
