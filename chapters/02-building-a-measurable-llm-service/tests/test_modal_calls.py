"""Exercise queue waits and cleanup without submitting remote work."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from modal_calls import invoke


def method(callback):
    return SimpleNamespace(aio=callback)


@pytest.mark.parametrize("failure", [None, "timeout", "remote", "interrupt", "cancel_timeout"])
def test_result_or_failure_preserves_call_and_requests_cleanup(tmp_path, failure):
    cancelled = []

    async def get(**kwargs):
        if failure in {"timeout", "cancel_timeout"}:
            await asyncio.sleep(10)
        if failure == "remote":
            raise RuntimeError("remote failure")
        if failure == "interrupt":
            raise asyncio.CancelledError()
        return {"result": 42}

    async def cancel(**kwargs):
        cancelled.append(kwargs)
        if failure == "cancel_timeout":
            await asyncio.sleep(10)

    async def spawn():
        return SimpleNamespace(object_id="fc-test", get=method(get), cancel=method(cancel))

    record = tmp_path / "call.json"
    coroutine = invoke(
        SimpleNamespace(spawn=method(spawn)), wait_s=0.05, cancel_s=0.05, record=record
    )
    if failure:
        expected = {
            "timeout": TimeoutError,
            "cancel_timeout": TimeoutError,
            "remote": RuntimeError,
            "interrupt": asyncio.CancelledError,
        }[failure]
        with pytest.raises(expected):
            asyncio.run(coroutine)
        assert cancelled == [{"terminate_containers": True}]
    else:
        assert asyncio.run(coroutine) == {"result": 42}
        assert cancelled == []
    saved = json.loads(record.read_text())
    assert saved["call_id"] == "fc-test"
    assert saved["cleanup_verified"] is False
    assert saved["status"] == ("failed" if failure else "returned")
    if failure == "cancel_timeout":
        assert saved["cancellation"] == "unverified: TimeoutError"


def test_submission_timeout_records_unknown_remote_state(tmp_path):
    async def spawn():
        await asyncio.sleep(10)

    record = tmp_path / "call.json"
    with pytest.raises(TimeoutError):
        asyncio.run(invoke(SimpleNamespace(spawn=method(spawn)), wait_s=0.01, record=record))
    saved = json.loads(record.read_text())
    assert saved["call_id"] is None
    assert saved["cleanup_verified"] is False
    assert saved["cancellation"] == "no call handle; stop and inspect the app"
