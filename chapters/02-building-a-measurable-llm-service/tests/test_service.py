"""Real loopback HTTP fixtures; no engine, credentials, or model downloads."""

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from pathlib import Path

import pytest
from analyze import analyze, summarize
from client import ChatClient
from experiment import run_condition, run_experiment
from records import RequestResult, RequestSpec, RunWriter
from server_process import running_server


def event(content=None, finish=None, usage=None):
    value = {
        "id": "fixture-response",
        "model": "fixture",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish}],
    }
    if usage:
        value["usage"] = usage
        value["choices"] = []
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\r\n\r\n").encode()


@asynccontextmanager
async def endpoint(mode="normal"):
    state = {"active": 0, "peak": 0, "calls": 0, "closed": 0}
    handlers = set()

    async def handle(reader, writer):
        handlers.add(asyncio.current_task())
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(
                (
                    int(h.split(b":")[1])
                    for h in headers.split(b"\r\n")
                    if h.lower().startswith(b"content-length:")
                ),
                0,
            )
            payload = json.loads(await reader.readexactly(length)) if length else {}
            if headers.startswith(b"GET"):
                body = json.dumps({"data": [{"id": "fixture"}]}).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: "
                    + str(len(body)).encode()
                    + b"\r\n\r\n"
                    + body
                )
                await writer.drain()
                return
            state["calls"] += 1
            if mode == "http_error":
                writer.write(b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            if not payload.get("stream"):
                body = json.dumps(
                    {
                        "id": "fixture-response",
                        "model": "fixture",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"content": "caf\u00e9"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                    }
                ).encode()
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: "
                    + str(len(body)).encode()
                    + b"\r\n\r\n"
                    + body
                )
                await writer.drain()
                return
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            if mode == "idle":
                await asyncio.sleep(0.3)
                return
            if mode == "trickle":
                for _ in range(30):
                    writer.write(b": heartbeat\n\n")
                    await writer.drain()
                    await asyncio.sleep(0.01)
                return
            if mode == "malformed":
                writer.write(b"data: {broken}\n\n")
            else:
                await asyncio.sleep(0.005)
                body = b": comment\r\n\r\n" + event("") + event("caf\u00e9")
                if mode != "disconnect":
                    body += event(finish="stop")
                    if mode != "missing_usage":
                        body += event(usage={"prompt_tokens": 10, "completion_tokens": 2})
                    body += b"data: [DONE]\r\n\r\n"
                # Force UTF-8 across reads; combine other events within one write.
                split = body.index("\u00e9".encode()) + 1
                writer.write(body[:split])
                await writer.drain()
                await asyncio.sleep(0.005)
                writer.write(body[split:])
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            state["active"] -= 1
            state["closed"] += 1
            handlers.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    try:
        yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1", state
    finally:
        server.close()
        await server.wait_closed()
        for task in list(handlers):
            task.cancel()
        await asyncio.gather(*list(handlers), return_exceptions=True)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("normal", "success"),
        ("missing_usage", "success"),
        ("http_error", "http_error"),
        ("malformed", "protocol_error"),
        ("disconnect", "protocol_error"),
        ("idle", "idle_timeout"),
        ("trickle", "deadline"),
    ],
)
def test_stream_boundaries_and_failures(mode, expected):
    async def exercise():
        async with (
            endpoint(mode) as (url, _state),
            ChatClient(url, "fixture", deadline_s=0.15, idle_timeout_s=0.08) as client,
        ):
            result = await client.generate(RequestSpec("r1", "p1", []), origin=time.perf_counter())
        assert result.status == expected
        if expected == "success":
            assert result.text == "caf\u00e9"
            assert result.first_content_s is not None
            assert result.output_tokens == (None if mode == "missing_usage" else 2)
        if mode == "disconnect":
            assert result.text == "caf\u00e9"
            assert result.output_check_passed is None

    asyncio.run(exercise())


def test_cancel_and_nonstream():
    async def exercise():
        async with endpoint() as (url, _state), ChatClient(url, "fixture") as client:
            result = await client.generate(
                RequestSpec("r1", "p1", [], cancel_after_content=True), origin=time.perf_counter()
            )
            assert result.status == "cancelled"
            assert result.engine_cancellation == "unverified"
            result = await client.generate(
                RequestSpec("r2", "p1", [], stream=False), origin=time.perf_counter()
            )
            assert result.status == "success"
            assert result.first_content_s is None

    asyncio.run(exercise())


def fixture_config():
    config = json.loads((Path(__file__).parents[1] / "experiment.json").read_text())
    config.update(
        model="fixture",
        request_deadline_s=0.5,
        idle_timeout_s=0.2,
        run_deadline_s=20,
        flush_reserve_s=0.1,
    )
    return config


def test_complete_experiment_and_analysis(tmp_path):
    async def exercise():
        async with endpoint() as (url, state):
            manifest = await run_experiment(
                url, fixture_config(), tmp_path / "run", {"evidence_kind": "SYNTHETIC HTTP FIXTURE"}
            )
            assert manifest["status"] == "completed", manifest["error"]
            assert manifest["generation_requests"] == state["calls"] == 291
            assert 2 <= state["peak"] <= 4
        summary = analyze(tmp_path / "run", plots=False)
        assert len(summary) == 6
        assert sum(s["requests"] for s in summary) == 144
        assert all(s["token_count_coverage"] == 1 for s in summary)

    asyncio.run(exercise())


def test_interrupted_workers_save_terminal_records(tmp_path):
    async def exercise():
        async with endpoint("idle") as (url, _state), ChatClient(url, "fixture") as client:
            writer = RunWriter(tmp_path / "partial", {"status": "incomplete"})
            task = asyncio.create_task(
                run_condition(
                    client,
                    [RequestSpec(f"r{i}", "p1", []) for i in range(20)],
                    writer,
                    time.perf_counter(),
                    2,
                    time.perf_counter() + 60,
                )
            )
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            writer.close()
        rows = [
            json.loads(s) for s in (tmp_path / "partial/requests.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 2
        assert all(row["status"] == "cancelled" for row in rows)

    asyncio.run(exercise())


def test_accounting_includes_failed_time_and_missing_counts():
    rows = []
    for index, (end, status, tokens) in enumerate(
        [(2, "success", 0), (4, "success", 1), (6, "success", None), (10, "deadline", None)]
    ):
        row = asdict(
            RequestResult(
                str(index),
                "p",
                "measure",
                0,
                1,
                True,
                0,
                ended_s=end,
                status=status,
                output_tokens=tokens,
                usage_source="server_reported" if tokens is not None else None,
                output_check_passed=index == 1,
            )
        )
        rows.append(row)
    result = summarize(rows, 5)[0]
    assert result["requests_per_s"] == 0.3
    assert result["failure_fraction"] == 0.25
    assert result["token_count_coverage"] == pytest.approx(2 / 3)
    assert result["counted_output_tokens_per_s"] == 0.1
    assert result["objective_requests_per_s"] == 0.1
    assert result["approx_output_interval_s_n"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rows + rows[:1], 5)


def test_server_startup_failure_cleans_process(tmp_path, monkeypatch):
    import subprocess

    captured = []
    popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        captured.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)

    async def exercise():
        with pytest.raises(TimeoutError):
            async with running_server(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                "http://127.0.0.1:1/v1",
                tmp_path / "server.log",
                readiness_s=0.5,
                shutdown_s=0.1,
            ):
                pytest.fail("Unready server must not be yielded")
        assert len(captured) == 1
        assert captured[0].poll() is not None

    asyncio.run(exercise())


def test_late_health_response_cannot_pass_readiness_deadline(tmp_path, monkeypatch):
    import subprocess

    import httpx

    processes = []
    popen = subprocess.Popen
    async_client = httpx.AsyncClient
    response_returned = False

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    async def slow_health(request):
        nonlocal response_returned
        await asyncio.sleep(0.2)
        response_returned = True
        return httpx.Response(200)

    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: async_client(transport=httpx.MockTransport(slow_health), **kwargs),
    )

    async def exercise():
        with pytest.raises(TimeoutError, match="Server readiness deadline expired"):
            async with running_server(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                "http://fixture/v1",
                tmp_path / "server.log",
                readiness_url="http://fixture/health",
                readiness_s=0.02,
                shutdown_s=0.5,
            ):
                pytest.fail("A late HTTP 200 must not establish readiness")
        assert not response_returned
        assert len(processes) == 1
        assert processes[0].poll() is not None

    asyncio.run(exercise())


def test_sse_multiline_bom_and_incomplete_event():
    from client import sse_data

    async def exercise():
        async def lines():
            for line in [
                "\ufeff: comment",
                "data: first",
                "data:second",
                "",
                "data: never dispatched",
            ]:
                yield line

        assert [item async for item in sse_data(lines())] == ["first\nsecond"]

    asyncio.run(exercise())


def test_same_text_and_tokens_with_different_chunking():
    import httpx

    async def exercise():
        results = []
        for chunks in [["hello"], ["h", "e", "l", "l", "o"]]:
            body = b"".join(event(c) for c in chunks) + event(finish="stop")
            body += event(usage={"completion_tokens": 1}) + b"data: [DONE]\n\n"
            async with ChatClient("http://fixture/v1", "fixture") as client:
                result = RequestResult("r1", "p1", "measure", 0, 1, True, 0)
                await client._stream(
                    httpx.Response(200, content=body),
                    result,
                    RequestSpec("r1", "p1", []),
                    time.perf_counter(),
                )
                results.append(result)
        assert [r.text for r in results] == ["hello", "hello"]
        assert [r.output_tokens for r in results] == [1, 1]
        assert len(results[0].events) != len(results[1].events)

    asyncio.run(exercise())


def test_partial_run_manifest_on_smoke_failure(tmp_path):
    async def exercise():
        async with endpoint("http_error") as (url, _state):
            manifest = await run_experiment(url, fixture_config(), tmp_path / "failed")
        assert manifest["status"] == "incomplete"
        assert manifest["generation_requests"] == 1
        assert "smoke failed" in manifest["error"]
        assert json.loads((tmp_path / "failed/manifest.json").read_text()) == manifest

    asyncio.run(exercise())


def test_prompt_limit_counts_tokens_not_batch_encoding_fields():
    from prepare_model import prompt_lengths
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    prompts = [{"id": "long", "messages": [{"role": "user", "content": "word word word"}]}]
    assert prompt_lengths(tokenizer, prompts, 3) == {"long": 3}
    with pytest.raises(ValueError, match="exceeds"):
        prompt_lengths(tokenizer, prompts, 2)


def test_server_parent_can_drain_child_before_escalation(tmp_path):
    """Signalling the whole group first would kill the worker before parent cleanup."""
    ready = tmp_path / "ready"
    drained = tmp_path / "drained"
    child = (
        "import signal,time; from pathlib import Path; "
        f'signal.signal(signal.SIGUSR1, lambda *_: (Path({str(drained)!r}).write_text("yes"), '
        "exit(0))); "
        f'Path({str(ready)!r}).write_text("ready"); time.sleep(60)'
    )
    parent = f"""
import signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', {child!r}])
def shutdown(*_):
    child.send_signal(signal.SIGUSR1)
    child.wait(timeout=2)
    sys.exit(0)
signal.signal(signal.SIGTERM, shutdown)
time.sleep(60)
"""

    async def exercise():
        async with endpoint() as (url, _state):
            with pytest.raises(RuntimeError, match="fixture body failed"):
                async with running_server(
                    [sys.executable, "-c", parent], url, tmp_path / "server.log", shutdown_s=2
                ):
                    async with asyncio.timeout(2):
                        while not ready.exists():
                            await asyncio.sleep(0.01)
                    raise RuntimeError("fixture body failed")
        assert drained.read_text() == "yes"

    asyncio.run(exercise())


def test_cancellation_during_shutdown_kills_and_reaps_server(tmp_path, monkeypatch):
    import os
    import signal
    import subprocess

    ready = tmp_path / "ready"
    command = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).touch(); time.sleep(60)"
    )

    async def exercise():
        terminating = asyncio.Event()
        captured = []
        terminate = subprocess.Popen.terminate

        def mark_termination(process):
            captured.append(process)
            terminate(process)
            terminating.set()

        monkeypatch.setattr(subprocess.Popen, "terminate", mark_termination)
        async with endpoint() as (url, _state):

            async def serve():
                async with running_server(
                    [sys.executable, "-c", command],
                    url,
                    tmp_path / "server.log",
                    shutdown_s=0.5,
                ):
                    async with asyncio.timeout(3):
                        while not ready.exists():
                            await asyncio.sleep(0.01)

            task = asyncio.create_task(serve())
            try:
                await asyncio.wait_for(terminating.wait(), timeout=5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert captured[0].returncode == -signal.SIGKILL
                with pytest.raises(ProcessLookupError):
                    os.killpg(captured[0].pid, 0)
            finally:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                for process in captured:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=2)

    asyncio.run(exercise())


def test_metadata_available_before_engine_readiness(tmp_path):
    async def exercise():
        checked = []

        async def handle(reader, writer):
            request = await reader.readuntil(b"\r\n\r\n")
            path = request.split()[1].decode()
            checked.append(path)
            status = b"503 Unavailable" if checked.count("/health") < 3 else b"200 OK"
            if path == "/v1/models":
                status = b"200 OK"
            writer.write(
                b"HTTP/1.1 " + status + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        try:
            async with running_server(
                [sys.executable, "-c", "import time;time.sleep(60)"],
                url + "/v1",
                tmp_path / "server.log",
                readiness_url=url + "/health",
                readiness_s=2,
            ):
                assert checked == ["/health", "/health", "/health"]
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(exercise())
