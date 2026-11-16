"""Save two synthetic lessons: events are not tokens, and failures occupy time.

The in-process HTTP transport supplies invented text and usage, not a model.
Elapsed times come from this execution; they say nothing about GPU performance.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import httpx
from analyze import summarize
from client import ChatClient
from evidence import validate_events
from records import RequestSpec, RunWriter, write_json


class FixtureStream(httpx.AsyncByteStream):
    def __init__(self, pieces: list[str], *, fail: bool = False):
        self.pieces = pieces
        self.fail = fail

    async def __aiter__(self):
        if self.fail:
            # The whole-request deadline ends this response; no retry follows.
            await asyncio.sleep(1)
        await asyncio.sleep(0.01)
        for piece in self.pieces:
            yield self.event([{"index": 0, "delta": {"content": piece}, "finish_reason": None}])
        yield self.event([{"index": 0, "delta": {}, "finish_reason": "stop"}])
        yield self.event([], usage={"prompt_tokens": 3, "completion_tokens": 1})
        yield b"data: [DONE]\n\n"

    @staticmethod
    def event(choices, **fields):
        payload = {"id": "synthetic-response", "model": "fixture", "choices": choices, **fields}
        return ("data: " + json.dumps(payload) + "\n\n").encode()


async def record_case(directory: Path, pieces: list[str], *, include_failure=False):
    def respond(request):
        payload = json.loads(request.content)
        fail = payload["messages"][0]["content"] == "fail"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=FixtureStream(pieces, fail=fail),
        )

    writer = RunWriter(
        directory,
        {
            "evidence_kind": "SYNTHETIC IN-PROCESS HTTP FIXTURE; no model or network",
            "status": "incomplete",
            "fixture_output_tokens": 1,
        },
    )
    results = []
    origin = time.perf_counter()
    try:
        async with ChatClient(
            "http://fixture/v1", "fixture", deadline_s=0.1, transport=httpx.MockTransport(respond)
        ) as client:
            for index, prompt in enumerate(["hello", "fail"] if include_failure else ["hello"], 1):
                result = await client.generate(
                    RequestSpec(
                        f"r{index:04}",
                        prompt,
                        [{"role": "user", "content": prompt}],
                        expected_text="hello",
                    ),
                    origin=origin,
                )
                writer.save(result)
                results.append(result)
    finally:
        writer.close(status="completed" if results else "incomplete")
    rows = [asdict(r) for r in results]
    events = [{"request_id": r.request_id, **e} for r in results for e in r.events]
    validate_events(rows, events)
    return rows, events


async def demonstrate(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    segmentation = []
    for name, pieces in [("one-event", ["hello"]), ("five-events", list("hello"))]:
        rows, events = await record_case(output / name, pieces)
        summary = summarize(rows, 1)[0]
        content_events = sum(
            e["kind"] == "sse"
            and any(c.get("delta", {}).get("content") for c in e["payload"]["choices"])
            for e in events
        )
        segmentation.append(
            {
                "case": name,
                "text": rows[0]["text"],
                "content_events": content_events,
                "fixture_reported_tokens": rows[0]["output_tokens"],
                "window_s": summary["window_s"],
                "wrong_events_as_tokens_per_s": content_events / summary["window_s"],
                "correct_fixture_tokens_per_s": summary["counted_output_tokens_per_s"],
            }
        )
    rows, _events = await record_case(output / "failure-mix", ["hello"], include_failure=True)
    complete = summarize(rows, 1)[0]
    omitted = summarize([r for r in rows if r["status"] == "success"], 1)[0]
    report = {
        "evidence_kind": "SYNTHETIC FIXTURE; timings are not model performance",
        "segmentation": segmentation,
        "failure_accounting": {
            "statuses": [r["status"] for r in rows],
            "correct_all_outcomes": complete,
            "wrong_failed_request_removed": omitted,
        },
        "limits": [
            "Token usage is defined by the fixture, not a tokenizer",
            "In-process HTTP transport, no network, engine or GPU",
            "Actual elapsed times vary; compare arithmetic within each recorded case",
        ],
    }
    write_json(output / "demonstrations.json", report)
    sources = output / "source"
    sources.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        (sources / path.name).write_bytes(path.read_bytes())
    write_json(
        output / "SHA256SUMS.json",
        {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(demonstrate(args.output)), indent=2))
