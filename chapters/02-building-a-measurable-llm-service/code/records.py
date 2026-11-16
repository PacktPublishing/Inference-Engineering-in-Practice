"""Small records shared by the HTTP client and offline analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt_id: str
    messages: list[dict[str, str]]
    phase: str = "measure"
    repeat: int = 0
    concurrency: int = 1
    stream: bool = True
    max_tokens: int = 64
    cancel_after_content: bool = False
    expected_text: str | None = None


@dataclass
class RequestResult:
    request_id: str
    prompt_id: str
    phase: str
    repeat: int
    concurrency: int
    stream: bool
    started_s: float
    ended_s: float = 0.0
    status: str = "pending"
    http_status: int | None = None
    response_id: str | None = None
    response_model: str | None = None
    text: str = ""
    finish_reason: str | None = None
    first_content_s: float | None = None
    last_content_s: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    usage_source: str | None = None
    output_check_passed: bool | None = None
    error: str | None = None
    engine_cancellation: str = "not_probed"
    events: list[dict[str, Any]] = field(default_factory=list)

    def event(self, kind: str, elapsed_s: float, **data: Any) -> None:
        self.events.append({"kind": kind, "elapsed_s": elapsed_s, **data})


def write_json(path: Path, value: Any) -> None:
    """Atomically replace metadata so an interrupted write is not a valid result."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def content_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class RunWriter:
    """Append terminal outcomes immediately; keep metadata for incomplete runs."""

    def __init__(self, directory: Path, manifest: dict[str, Any]):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.manifest = manifest
        self.requests = (directory / "requests.jsonl").open("w", buffering=1)
        self.events = (directory / "events.jsonl").open("w", buffering=1)
        write_json(directory / "manifest.json", manifest)

    def save(self, result: RequestResult) -> None:
        row = asdict(result)
        for event in row.pop("events"):
            self.events.write(json.dumps({"request_id": result.request_id, **event}) + "\n")
        self.events.flush()
        self.requests.write(json.dumps(row, allow_nan=False) + "\n")
        self.requests.flush()

    def close(self, **updates: Any) -> None:
        self.requests.close()
        self.events.close()
        self.manifest.update(updates)
        write_json(self.directory / "manifest.json", self.manifest)
