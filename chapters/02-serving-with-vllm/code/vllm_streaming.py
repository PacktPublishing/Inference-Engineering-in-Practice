"""Pure helpers for OpenAI-compatible server-sent event streams."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TimedPayload:
    """One decoded response payload and its client receive timestamp."""

    received_at: float
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StreamResult:
    """Assembled text and client-observed stream timings."""

    text: str
    ttft_seconds: float | None
    e2e_seconds: float
    content_events: int
    finish_reason: str | None
    usage: Mapping[str, Any] | None


def iter_sse_data(lines: Iterable[str | bytes]) -> Iterator[str]:
    """Yield complete SSE `data` fields, including multi-line events."""
    data_lines: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        line = line.rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue

        field, separator, value = line.partition(":")
        if field != "data":
            continue
        if separator and value.startswith(" "):
            value = value[1:]
        data_lines.append(value)

    if data_lines:
        yield "\n".join(data_lines)


def iter_openai_payloads(lines: Iterable[str | bytes]) -> Iterator[dict[str, Any]]:
    """Decode OpenAI-compatible JSON payloads until the `[DONE]` sentinel."""
    for data in iter_sse_data(lines):
        if data == "[DONE]":
            return
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("stream payload must be a JSON object")
        yield payload


def _first_choice(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    choices = payload.get("choices")
    if choices is None:
        return None
    if not isinstance(choices, list) or not choices:
        raise ValueError("choices must be a non-empty list when present")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("each choice must be an object")
    return choice


def content_delta(payload: Mapping[str, Any]) -> str:
    """Return text content while ignoring role-only and empty stream chunks."""
    choice = _first_choice(payload)
    if choice is None:
        return ""
    delta = choice.get("delta")
    if delta is None:
        return ""
    if not isinstance(delta, Mapping):
        raise ValueError("choice.delta must be an object")
    content = delta.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError("choice.delta.content must be text")
    return content


def summarize_stream(
    events: Iterable[TimedPayload],
    *,
    started_at: float,
    completed_at: float | None = None,
) -> StreamResult:
    """Assemble a stream without assuming one event equals one model token."""
    pieces: list[str] = []
    first_content_at: float | None = None
    finish_reason: str | None = None
    usage: Mapping[str, Any] | None = None
    previous_time = started_at
    last_event_at = started_at

    for event in events:
        if event.received_at < previous_time:
            raise ValueError("stream event timestamps must be monotonic")
        previous_time = event.received_at
        last_event_at = event.received_at

        text = content_delta(event.payload)
        if text:
            if first_content_at is None:
                first_content_at = event.received_at
            pieces.append(text)

        choice = _first_choice(event.payload)
        if choice is not None and choice.get("finish_reason") is not None:
            finish_reason = str(choice["finish_reason"])

        payload_usage = event.payload.get("usage")
        if payload_usage is not None:
            if not isinstance(payload_usage, Mapping):
                raise ValueError("usage must be an object")
            usage = payload_usage

    effective_completed_at = last_event_at if completed_at is None else completed_at
    if effective_completed_at < last_event_at:
        raise ValueError("completed_at cannot precede the last stream event")

    return StreamResult(
        text="".join(pieces),
        ttft_seconds=(
            None if first_content_at is None else first_content_at - started_at
        ),
        e2e_seconds=effective_completed_at - started_at,
        content_events=len(pieces),
        finish_reason=finish_reason,
        usage=usage,
    )
