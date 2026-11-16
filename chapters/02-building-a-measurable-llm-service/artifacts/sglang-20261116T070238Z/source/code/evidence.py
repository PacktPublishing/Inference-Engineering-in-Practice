"""Check saved records against protocol events and the declared finite workload."""

from __future__ import annotations

import math
from collections import defaultdict

from client import apply_usage
from records import RequestResult, content_hash

TERMINAL_STATUSES = {
    "success",
    "http_error",
    "idle_timeout",
    "transport_timeout",
    "deadline",
    "protocol_error",
    "transport_error",
    "cancelled",
}


def validate_events(rows, events):
    """Cross-check fields used by analysis; this does not authenticate a client clock."""
    grouped = defaultdict(list)
    for event in events:
        grouped[event["request_id"]].append(event)
    ids = [r["request_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate request ID")
    if set(grouped) != set(ids):
        raise ValueError("Request/event IDs disagree")
    for row in rows:
        trace = grouped[row["request_id"]]
        if trace[0]["kind"] != "submitted" or trace[-1]["kind"] != "terminal":
            raise ValueError("Incomplete event trace")
        times = [e["elapsed_s"] for e in trace]
        if any(not math.isfinite(t) or t < 0 for t in times) or times != sorted(times):
            raise ValueError("Invalid event chronology")
        if times[0] != 0 or sum(e["kind"] == "terminal" for e in trace) != 1:
            raise ValueError("Invalid submission or terminal event")
        start, end = row["started_s"], row["ended_s"]
        if not all(math.isfinite(t) for t in (start, end)) or start < 0 or end < start:
            raise ValueError("Invalid request times")
        if not math.isclose(end - start, times[-1], rel_tol=0, abs_tol=1e-9):
            raise ValueError("Terminal elapsed time mismatch")
        if row["status"] not in TERMINAL_STATUSES or trace[-1]["status"] != row["status"]:
            raise ValueError("Terminal status mismatch")
        headers = [e["http_status"] for e in trace if e["kind"] == "headers"]
        if len(headers) > 1 or (headers[0] if headers else None) != row["http_status"]:
            raise ValueError("HTTP status mismatch")
        # Invalid final payloads can be retained before parsing fails. Their partial
        # fields are diagnostic only and never contribute to successful throughput.
        if row["status"] == "protocol_error":
            continue
        rebuilt = RequestResult("check", "check", "measure", 0, 1, row["stream"], 0)
        done = False
        responses = 0
        for event in trace:
            kind = event["kind"]
            if kind == "done":
                if done or rebuilt.finish_reason is None:
                    raise ValueError("Invalid terminal marker")
                done = True
            if kind not in ("sse", "response"):
                continue
            if done:
                raise ValueError("Response payload after terminal marker")
            payload = event["payload"]
            for key, attr in (("id", "response_id"), ("model", "response_model")):
                value = payload.get(key)
                if value is not None:
                    if getattr(rebuilt, attr) not in (None, value):
                        raise ValueError("Response identity changed")
                    setattr(rebuilt, attr, value)
            apply_usage(rebuilt, payload.get("usage"))
            if kind == "response":
                responses += 1
                choice = payload["choices"][0]
                rebuilt.text = choice["message"]["content"]
                rebuilt.finish_reason = choice["finish_reason"]
                continue
            for choice in payload.get("choices", []):
                text = choice.get("delta", {}).get("content")
                if text:
                    if rebuilt.finish_reason is not None:
                        raise ValueError("Content after finish reason")
                    rebuilt.text += text
                    if rebuilt.first_content_s is None:
                        rebuilt.first_content_s = event["elapsed_s"]
                    rebuilt.last_content_s = event["elapsed_s"]
                reason = choice.get("finish_reason")
                if reason is not None:
                    rebuilt.finish_reason = reason
        if row["status"] == "success":
            if row["http_status"] != 200 or not rebuilt.finish_reason:
                raise ValueError("Successful response lacks HTTP success or finish reason")
            if row["stream"] and not done:
                raise ValueError("Successful stream lacks terminal marker")
            if not row["stream"] and responses != 1:
                raise ValueError("Successful ordinary response lacks response event")
        for field in (
            "text",
            "first_content_s",
            "last_content_s",
            "prompt_tokens",
            "output_tokens",
            "usage_source",
            "finish_reason",
            "response_id",
            "response_model",
        ):
            if row[field] != getattr(rebuilt, field):
                raise ValueError(f"Recorded {field} differs from event trace")


def expected_schedule(config):
    """Describe the teaching workload independently of its concurrent execution."""
    entries = []

    def add(phase, repeat, concurrency, prompts, stream=True):
        for prompt in prompts:
            entries.append(
                {
                    "phase": phase,
                    "repeat": repeat,
                    "concurrency": concurrency,
                    "prompt": prompt,
                    "stream": stream,
                }
            )

    for stream in (False, True):
        add("smoke", 0, 1, config["prompts"][:1], stream)
    for repeat in range(config["repeats"]):
        for concurrency in config["concurrency"][:: 1 if repeat % 2 == 0 else -1]:
            warmup = [
                config["prompts"][i % len(config["prompts"])]
                for i in range(config["warmup_requests"])
            ]
            add("warmup", repeat, concurrency, warmup)
            add("measure", repeat, concurrency, config["prompts"])
    add(
        "probe",
        0,
        1,
        [
            {
                "id": "cancellation",
                "messages": [
                    {
                        "role": "user",
                        "content": "List the integers from 1 to 200, separated by spaces.",
                    }
                ],
            }
        ],
    )
    return {f"r{i:04}": entry for i, entry in enumerate(entries, 1)}


def validate_run(manifest, rows, events):
    """Validate a complete or explicitly partial experiment without dropping failures."""
    validate_events(rows, events)
    config = manifest["config"]
    if manifest["config_sha256"] != content_hash(config):
        raise ValueError("Configuration hash mismatch")
    expected = expected_schedule(config)
    schedule = manifest["schedule"]
    if schedule != list(expected)[: len(schedule)]:
        raise ValueError("Declared schedule IDs differ from workload")
    if manifest["generation_requests"] != len(rows):
        raise ValueError("Manifest and terminal record counts disagree")
    actual_ids = {r["request_id"] for r in rows}
    if not actual_ids.issubset(schedule):
        raise ValueError("Unscheduled request record")
    if manifest["status"] == "completed" and (
        set(schedule) != set(expected) or actual_ids != set(expected)
    ):
        raise ValueError("Completed run has an incomplete schedule")
    submitted = {e["request_id"]: e["payload"] for e in events if e["kind"] == "submitted"}
    for row in rows:
        entry = expected[row["request_id"]]
        for field in ("phase", "repeat", "concurrency", "stream"):
            if row[field] != entry[field]:
                raise ValueError(f"Observed schedule differs: {field}")
        prompt = entry["prompt"]
        if row["prompt_id"] != prompt["id"]:
            raise ValueError("Observed prompt ID differs")
        payload = submitted[row["request_id"]]
        wanted = {
            "model": config["model"],
            "messages": prompt["messages"],
            "n": 1,
            "temperature": 0,
            "max_tokens": 512 if row["phase"] == "probe" else config["max_tokens"],
            "stream": entry["stream"],
        }
        if entry["stream"]:
            wanted["stream_options"] = {"include_usage": True}
        if payload != wanted:
            raise ValueError("Submitted request differs from workload")
        check = None
        if row["status"] == "success":
            expected_text = prompt.get("expected_text")
            check = (
                row["text"].strip() == expected_text.strip()
                if expected_text is not None
                else bool(row["text"].strip())
            )
            if row["response_model"] != config["model"]:
                raise ValueError("Response model differs from workload")
        if row["output_check_passed"] != check:
            raise ValueError("Recorded output check differs from saved text")
        if manifest["status"] == "completed" and row["phase"] in ("smoke", "warmup"):
            if row["status"] != "success":
                raise ValueError("Completed run has failed contract checks or warmup")
            if row["phase"] == "smoke" and (
                row["prompt_tokens"] is None or row["output_tokens"] is None
            ):
                raise ValueError("Contract checks lack token usage")
