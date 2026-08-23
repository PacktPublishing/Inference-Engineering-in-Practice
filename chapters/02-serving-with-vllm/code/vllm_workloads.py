"""Deterministic text-generation workloads for Chapter 2 experiments."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from vllm_config import RANDOM_SEED, WORKLOAD_SCHEMA_VERSION

ArrivalMode = Literal["closed_loop", "open_loop"]


@dataclass(frozen=True, slots=True)
class WorkloadRequest:
    """One model-independent chat request."""

    request_id: str
    user_prompt: str
    max_tokens: int
    system_prompt: str | None = None
    temperature: float = 0.0
    seed: int = RANDOM_SEED
    group: str = "default"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if not self.user_prompt:
            raise ValueError("user_prompt cannot be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not self.group:
            raise ValueError("group cannot be empty")

    def messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.user_prompt})
        return messages

    def api_payload(self, *, model: str, stream: bool = True) -> dict:
        if not model:
            raise ValueError("model cannot be empty")
        payload = {
            "model": model,
            "messages": self.messages(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        return payload


@dataclass(frozen=True, slots=True)
class Workload:
    """A reproducible group of requests and its arrival policy."""

    name: str
    requests: tuple[WorkloadRequest, ...]
    arrival_mode: ArrivalMode = "closed_loop"
    max_concurrency: int = 1
    request_rate: float | None = None
    schema_version: int = WORKLOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name cannot be empty")
        if not self.requests:
            raise ValueError("requests cannot be empty")
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request IDs must be unique")
        if self.arrival_mode not in {"closed_loop", "open_loop"}:
            raise ValueError("arrival_mode must be closed_loop or open_loop")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        if self.schema_version <= 0:
            raise ValueError("schema_version must be positive")
        if self.arrival_mode == "closed_loop" and self.request_rate is not None:
            raise ValueError("closed-loop workloads cannot set request_rate")
        if self.arrival_mode == "open_loop" and (
            self.request_rate is None or self.request_rate <= 0
        ):
            raise ValueError("open-loop workloads require a positive request_rate")


def make_concurrency_workload(
    *,
    count: int,
    max_concurrency: int,
    output_tokens: Sequence[int] = (32,),
) -> Workload:
    """Create deterministic requests with controlled output-token limits."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not output_tokens or any(token_count <= 0 for token_count in output_tokens):
        raise ValueError("output_tokens must contain positive values")
    requests = tuple(
        WorkloadRequest(
            request_id=f"concurrency-{index:04d}",
            user_prompt=(
                "Explain one practical consequence of KV-cache growth "
                f"for serving request {index}."
            ),
            max_tokens=output_tokens[index % len(output_tokens)],
            group=f"output-{output_tokens[index % len(output_tokens)]}",
        )
        for index in range(count)
    )
    return Workload(
        name="concurrency-sweep",
        requests=requests,
        max_concurrency=max_concurrency,
    )


def make_prefix_workload(
    *,
    shared_system_prompt: str,
    questions: Sequence[str],
    max_tokens: int = 32,
    group: str = "shared-prefix",
) -> Workload:
    """Create requests that share an exact system-prompt token prefix."""
    if not shared_system_prompt:
        raise ValueError("shared_system_prompt cannot be empty")
    if not questions or any(not question for question in questions):
        raise ValueError("questions must contain non-empty prompts")
    requests = tuple(
        WorkloadRequest(
            request_id=f"{group}-{index:04d}",
            system_prompt=shared_system_prompt,
            user_prompt=question,
            max_tokens=max_tokens,
            group=group,
        )
        for index, question in enumerate(questions)
    )
    return Workload(name=group, requests=requests, max_concurrency=1)


def workload_document(workload: Workload) -> dict:
    """Return the stable JSON representation written beside benchmark artifacts."""
    return {
        "schema_version": workload.schema_version,
        "name": workload.name,
        "arrival_mode": workload.arrival_mode,
        "max_concurrency": workload.max_concurrency,
        "request_rate": workload.request_rate,
        "requests": [asdict(request) for request in workload.requests],
    }


def write_workload_jsonl(
    workload: Workload,
    destination: str | Path,
    *,
    model: str,
) -> None:
    """Write one self-describing request record per JSONL line."""
    if not model:
        raise ValueError("model cannot be empty")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for request in workload.requests:
            record = {
                "schema_version": workload.schema_version,
                "workload": workload.name,
                "arrival_mode": workload.arrival_mode,
                "max_concurrency": workload.max_concurrency,
                "request_rate": workload.request_rate,
                "request_id": request.request_id,
                "group": request.group,
                "payload": request.api_payload(model=model),
            }
            file.write(json.dumps(record, sort_keys=True))
            file.write("\n")


def iter_request_ids(workloads: Iterable[Workload]) -> tuple[str, ...]:
    """Collect request IDs and reject collisions across workload files."""
    request_ids = tuple(
        request.request_id for workload in workloads for request in workload.requests
    )
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request IDs must be unique across workloads")
    return request_ids
