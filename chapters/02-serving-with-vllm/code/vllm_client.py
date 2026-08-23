"""Small OpenAI-compatible HTTP client used by Chapter 2 experiments."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx
from vllm_streaming import (
    StreamResult,
    TimedPayload,
    iter_openai_payloads,
    summarize_stream,
)


class VLLMClientError(RuntimeError):
    """Raised when a server response does not satisfy the chapter contract."""


@dataclass(frozen=True, slots=True)
class ClientTimeouts:
    """Explicit network timeout phases for reproducible client behavior."""

    connect: float = 10.0
    read: float = 120.0
    write: float = 30.0
    pool: float = 10.0

    def __post_init__(self) -> None:
        if min(self.connect, self.read, self.write, self.pool) <= 0:
            raise ValueError("all timeout values must be positive")

    def as_httpx(self) -> httpx.Timeout:
        return httpx.Timeout(
            connect=self.connect,
            read=self.read,
            write=self.write,
            pool=self.pool,
        )


class OpenAICompatibleClient:
    """Synchronous client with injectable transport and clock for local tests."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeouts: ClientTimeouts | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not base_url:
            raise ValueError("base_url cannot be empty")
        headers = {"Accept": "application/json"}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=(ClientTimeouts() if timeouts is None else timeouts).as_httpx(),
            transport=transport,
        )
        self._clock = clock

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as error:
            raise VLLMClientError("server returned invalid JSON") from error
        if not isinstance(payload, dict):
            raise VLLMClientError("server response must be a JSON object")
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            if not response.is_stream_consumed:
                response.read()
            body = response.text[:500]
            raise VLLMClientError(
                f"server returned HTTP {response.status_code}: {body}"
            ) from error

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(method, path, json=json_body)
        except httpx.HTTPError as error:
            raise VLLMClientError(f"{method} {path} failed: {error}") from error
        self._raise_for_status(response)
        return self._json_object(response)

    def list_models(self) -> tuple[str, ...]:
        payload = self._request_json("GET", "/v1/models")
        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise VLLMClientError("model list response must contain a data array")
        model_ids: list[str] = []
        for model in raw_models:
            if not isinstance(model, Mapping) or not isinstance(model.get("id"), str):
                raise VLLMClientError("each model entry must contain a text id")
            model_ids.append(model["id"])
        return tuple(model_ids)

    def complete(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        request["stream"] = False
        request.pop("stream_options", None)
        return self._request_json(
            "POST",
            "/v1/chat/completions",
            json_body=request,
        )

    def stream(self, payload: Mapping[str, Any]) -> StreamResult:
        request = dict(payload)
        request["stream"] = True
        request.setdefault("stream_options", {"include_usage": True})
        started_at = self._clock()
        events: list[TimedPayload] = []

        try:
            with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=request,
                headers={"Accept": "text/event-stream"},
            ) as response:
                self._raise_for_status(response)
                for stream_payload in iter_openai_payloads(response.iter_lines()):
                    events.append(
                        TimedPayload(
                            received_at=self._clock(),
                            payload=stream_payload,
                        )
                    )
            return summarize_stream(
                events,
                started_at=started_at,
                completed_at=self._clock(),
            )
        except VLLMClientError:
            raise
        except (httpx.HTTPError, UnicodeError, ValueError) as error:
            raise VLLMClientError(f"stream request failed: {error}") from error
