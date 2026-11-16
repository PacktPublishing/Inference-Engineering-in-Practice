"""A time-bounded Chat Completions client; HTTP events are not model tokens.

Stream size guards limit dispatched data payloads and retained events. They do
not bound all incoming framing/comment bytes or HTTPX's unfinished line buffer.
This teaching client targets the selected local engines, not hostile endpoints.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from records import RequestResult, RequestSpec

MAX_EVENT_CHARS = 262_144
MAX_RESPONSE_BYTES = 2_097_152
MAX_EVENTS = 4096


class ProtocolError(ValueError):
    """The response cannot satisfy this chapter's single text-choice contract."""


async def sse_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Frame data fields after HTTPX's incremental UTF-8 and newline decoding.

    A blank line dispatches an event. Unterminated data at EOF is not an event.
    Comments and fields unrelated to this one-shot stream are ignored; no reconnect.
    """
    fields: list[str] = []
    size = 0
    first = True
    async for line in lines:
        if first:
            line = line.removeprefix("\ufeff")
            first = False
        size += len(line)
        if size > MAX_EVENT_CHARS:
            raise ProtocolError("SSE event exceeds the chapter limit")
        if line == "":
            if fields:
                yield "\n".join(fields)
            fields, size = [], 0
        elif line.startswith("data:"):
            value = line[5:]
            fields.append(value[1:] if value.startswith(" ") else value)
        elif line == "data":
            fields.append("")


def object_json(data: str | bytes) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ProtocolError("Expected a JSON object")
    if "error" in value:
        raise ProtocolError("Server returned an error object")
    return value


def apply_usage(result: RequestResult, usage: Any) -> None:
    if usage is None:
        return
    if not isinstance(usage, dict):
        raise ProtocolError("Invalid usage object")
    for name, attribute in [
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "output_tokens"),
    ]:
        value = usage.get(name)
        if value is not None and (type(value) is not int or value < 0):
            raise ProtocolError(f"Invalid {name}")
        if value is not None:
            previous = getattr(result, attribute)
            if previous is not None and previous != value:
                raise ProtocolError("Conflicting usage counts")
            setattr(result, attribute, value)
            result.usage_source = "server_reported"


def choice_zero(payload: dict[str, Any], *, allow_empty: bool) -> dict[str, Any] | None:
    choices = payload.get("choices")
    if choices == [] and allow_empty and payload.get("usage") is not None:
        return None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ProtocolError("Expected one choice (or a usage-only event)")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("index") != 0:
        raise ProtocolError("Expected choice index zero")
    return choice


class ChatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        deadline_s: float = 30,
        idle_timeout_s: float = 10,
        max_connections: int = 4,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if deadline_s <= 0 or idle_timeout_s <= 0 or max_connections < 1:
            raise ValueError("Timeouts and connection limit must be positive")
        self.model = model
        self.deadline_s = deadline_s
        self.http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(idle_timeout_s),
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_connections
            ),
            trust_env=False,
            follow_redirects=False,
            transport=transport,
        )

    async def __aenter__(self) -> ChatClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.http.aclose()

    async def generate(self, spec: RequestSpec, *, origin: float) -> RequestResult:
        started = time.perf_counter()
        result = RequestResult(
            spec.request_id,
            spec.prompt_id,
            spec.phase,
            spec.repeat,
            spec.concurrency,
            spec.stream,
            started - origin,
        )
        if not 1 <= spec.max_tokens <= 512:
            raise ValueError("max_tokens must be between 1 and 512")
        payload = {
            "model": self.model,
            "messages": spec.messages,
            "n": 1,
            "temperature": 0,
            "max_tokens": spec.max_tokens,
            "stream": spec.stream,
        }
        if spec.stream:
            payload["stream_options"] = {"include_usage": True}
        result.event("submitted", 0.0, payload=payload)
        try:
            async with asyncio.timeout(self.deadline_s):
                async with self.http.stream(
                    "POST",
                    "chat/completions",
                    json=payload,
                    headers={"X-Request-Id": spec.request_id},
                ) as response:
                    result.http_status = response.status_code
                    result.event(
                        "headers", time.perf_counter() - started, http_status=response.status_code
                    )
                    response.raise_for_status()
                    if spec.stream:
                        if response.headers.get("content-type", "").split(";")[0] != (
                            "text/event-stream"
                        ):
                            raise ProtocolError("Expected text/event-stream")
                        await self._stream(response, result, spec, started)
                    else:
                        body = bytearray()
                        async for part in response.aiter_bytes():
                            body.extend(part)
                            if len(body) > MAX_RESPONSE_BYTES:
                                raise ProtocolError("Response exceeds the chapter limit")
                        data = object_json(body)
                        choice = choice_zero(data, allow_empty=False)
                        message = choice.get("message")
                        if not isinstance(message, dict):
                            raise ProtocolError("Missing message")
                        content = message.get("content")
                        if not isinstance(content, str):
                            raise ProtocolError("Expected a text response")
                        result.text = content
                        result.finish_reason = choice.get("finish_reason")
                        if not isinstance(result.finish_reason, str) or not result.finish_reason:
                            raise ProtocolError("Missing finish reason")
                        result.response_id = data.get("id")
                        result.response_model = data.get("model")
                        apply_usage(result, data.get("usage"))
                        result.event("response", time.perf_counter() - started, payload=data)
                        result.status = "success"
        except httpx.HTTPStatusError:
            result.status, result.error = "http_error", f"HTTP {result.http_status}"
        except httpx.ReadTimeout:
            result.status, result.error = "idle_timeout", "No response bytes within idle timeout"
        except httpx.TimeoutException:
            result.status, result.error = "transport_timeout", "Connect/write/pool timeout"
        except TimeoutError:
            result.status, result.error = "deadline", "Whole-request deadline expired"
        except (ProtocolError, json.JSONDecodeError, UnicodeError, httpx.DecodingError) as exc:
            result.status, result.error = "protocol_error", str(exc)
        except httpx.TransportError as exc:
            result.status, result.error = "transport_error", type(exc).__name__
        except asyncio.CancelledError:
            result.status, result.error = "cancelled", "Run interrupted; client stream closed"
            result.engine_cancellation = "unverified"
        result.ended_s = time.perf_counter() - origin
        if result.status == "success":
            result.output_check_passed = (
                result.text.strip() == spec.expected_text.strip()
                if spec.expected_text is not None
                else bool(result.text.strip())
            )
        result.event("terminal", result.ended_s - result.started_s, status=result.status)
        return result

    async def _stream(
        self, response: httpx.Response, result: RequestResult, spec: RequestSpec, started: float
    ) -> None:
        total = 0
        async for data in sse_data(response.aiter_lines()):
            elapsed = time.perf_counter() - started
            total += len(data.encode())
            if total > MAX_RESPONSE_BYTES or len(result.events) >= MAX_EVENTS:
                raise ProtocolError("Stream exceeds the chapter limit")
            if data == "[DONE]":
                result.event("done", elapsed)
                if result.finish_reason is None:
                    raise ProtocolError("Terminal marker without a finish reason")
                result.status = "success"
                return
            payload = object_json(data)
            result.event("sse", elapsed, payload=payload)
            for key, attribute in [("id", "response_id"), ("model", "response_model")]:
                value = payload.get(key)
                if value is not None:
                    if getattr(result, attribute) not in (None, value):
                        raise ProtocolError(f"Response {key} changed during stream")
                    setattr(result, attribute, value)
            apply_usage(result, payload.get("usage"))
            choice = choice_zero(payload, allow_empty=True)
            if choice is None:
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                raise ProtocolError("Missing text delta")
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                raise ProtocolError("Invalid text delta")
            if content:
                if result.finish_reason is not None:
                    raise ProtocolError("Content after finish reason")
                result.text += content
                if result.first_content_s is None:
                    result.first_content_s = elapsed
                result.last_content_s = elapsed
            reason = choice.get("finish_reason")
            if reason is not None:
                if not isinstance(reason, str) or not reason or result.finish_reason is not None:
                    raise ProtocolError("Invalid or repeated finish reason")
                result.finish_reason = reason
            if content and spec.cancel_after_content:
                result.status = "cancelled"
                result.error = "Deliberate probe: closed after first content"
                result.engine_cancellation = "unverified"
                return
        raise ProtocolError("Stream ended without [DONE]")
