import json

import httpx
import pytest
from vllm_client import (
    ClientTimeouts,
    OpenAICompatibleClient,
    VLLMClientError,
)


def test_model_discovery_and_non_streaming_completion() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "served-model"}]})
        body = json.loads(request.content)
        assert body["stream"] is False
        assert "stream_options" not in body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}]},
        )

    with OpenAICompatibleClient(
        "https://example.invalid/",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    ) as client:
        assert client.list_models() == ("served-model",)
        response = client.complete(
            {
                "model": "served-model",
                "messages": [{"role": "user", "content": "Question"}],
                "stream_options": {"include_usage": True},
            }
        )

    assert response["choices"][0]["message"]["content"] == "answer"
    assert requests[0].headers["authorization"] == "Bearer test-key"


def test_streaming_measures_first_content_not_role_chunk() -> None:
    stream_body = "".join(
        [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            (
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":2,"completion_tokens":1}}\n\n'
            ),
            "data: [DONE]\n\n",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "text/event-stream"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream_body,
        )

    times = iter([100.0, 100.1, 100.3, 100.4, 100.5])
    with OpenAICompatibleClient(
        "https://example.invalid",
        transport=httpx.MockTransport(handler),
        clock=lambda: next(times),
    ) as client:
        result = client.stream(
            {
                "model": "served-model",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )

    assert result.text == "Hello"
    assert result.ttft_seconds == pytest.approx(0.3)
    assert result.e2e_seconds == pytest.approx(0.5)
    assert result.content_events == 1


def test_http_errors_include_status_and_bounded_body() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, text="server unavailable")
    )

    with OpenAICompatibleClient(
        "https://example.invalid",
        transport=transport,
    ) as client, pytest.raises(VLLMClientError, match="HTTP 503.*server unavailable"):
        client.list_models()


def test_streaming_http_and_payload_errors_use_the_client_error_contract() -> None:
    error_transport = httpx.MockTransport(
        lambda request: httpx.Response(503, text="stream unavailable")
    )
    with (
        OpenAICompatibleClient(
            "https://example.invalid",
            transport=error_transport,
        ) as client,
        pytest.raises(VLLMClientError, match="HTTP 503.*stream unavailable"),
    ):
        client.stream({"model": "served-model", "messages": []})

    invalid_transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: not-json\n\n",
        )
    )
    with (
        OpenAICompatibleClient(
            "https://example.invalid",
            transport=invalid_transport,
        ) as client,
        pytest.raises(VLLMClientError, match="stream request failed"),
    ):
        client.stream({"model": "served-model", "messages": []})


def test_invalid_timeout_and_model_response_are_rejected() -> None:
    with pytest.raises(ValueError, match="timeout"):
        ClientTimeouts(read=0)

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [{"name": "missing-id"}]})
    )
    with OpenAICompatibleClient(
        "https://example.invalid",
        transport=transport,
    ) as client, pytest.raises(VLLMClientError, match="text id"):
        client.list_models()
