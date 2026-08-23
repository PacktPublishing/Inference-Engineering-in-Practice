import json

import pytest
from vllm_streaming import (
    TimedPayload,
    iter_openai_payloads,
    iter_sse_data,
    summarize_stream,
)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def test_sse_parser_handles_comments_multiline_events_and_done() -> None:
    lines = [
        ": keep-alive\n",
        "data: first\n",
        "data: second\n",
        "\n",
        b"data: [DONE]\n",
        b"\n",
        "data: ignored\n",
        "\n",
    ]

    assert list(iter_sse_data(lines)) == ["first\nsecond", "[DONE]", "ignored"]
    stream_lines = (
        _sse({"id": "one"}) + "data: [DONE]\n\n"
    ).splitlines(keepends=True)
    assert list(iter_openai_payloads(stream_lines)) == [{"id": "one"}]


def test_stream_summary_ignores_role_only_chunks_and_uses_first_content() -> None:
    payloads = [
        TimedPayload(
            10.1,
            {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        ),
        TimedPayload(
            10.3,
            {"choices": [{"delta": {"content": "two words"}, "finish_reason": None}]},
        ),
        TimedPayload(
            10.5,
            {"choices": [{"delta": {"content": " together"}, "finish_reason": None}]},
        ),
        TimedPayload(
            10.6,
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        ),
    ]

    result = summarize_stream(payloads, started_at=10.0, completed_at=10.7)

    assert result.text == "two words together"
    assert result.ttft_seconds == pytest.approx(0.3)
    assert result.e2e_seconds == pytest.approx(0.7)
    assert result.content_events == 2
    assert result.finish_reason == "stop"
    assert result.usage == {"prompt_tokens": 4, "completion_tokens": 2}


def test_stream_summary_does_not_require_content_for_control_events() -> None:
    result = summarize_stream(
        [TimedPayload(1.1, {"choices": [{"delta": {}, "finish_reason": "length"}]})],
        started_at=1.0,
    )

    assert result.text == ""
    assert result.ttft_seconds is None
    assert result.finish_reason == "length"


def test_stream_summary_rejects_non_monotonic_timestamps() -> None:
    events = [
        TimedPayload(2.0, {"choices": [{"delta": {"content": "a"}}]}),
        TimedPayload(1.5, {"choices": [{"delta": {"content": "b"}}]}),
    ]

    with pytest.raises(ValueError, match="monotonic"):
        summarize_stream(events, started_at=1.0)


def test_openai_payloads_reject_non_object_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        list(iter_openai_payloads(["data: []\n\n"]))
