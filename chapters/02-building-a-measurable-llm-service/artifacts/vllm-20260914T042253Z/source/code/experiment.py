"""One fixed closed-loop experiment, usable against any compatible local endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from client import ChatClient
from records import RequestSpec, RunWriter, content_hash


def validate(config: dict) -> None:
    if not config["prompts"] or len({p["id"] for p in config["prompts"]}) != len(config["prompts"]):
        raise ValueError("Prompt IDs must be unique and nonempty")
    for key in ("repeats", "warmup_requests", "request_cap", "max_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["max_tokens"] > 512 or config["request_cap"] > 180:
        raise ValueError("Chapter request/output cap exceeded")
    if not config["concurrency"] or any(type(c) is not int or c < 1 for c in config["concurrency"]):
        raise ValueError("Invalid concurrency")
    for key in (
        "request_deadline_s",
        "idle_timeout_s",
        "run_deadline_s",
        "flush_reserve_s",
        "objective_latency_s",
    ):
        if config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    required = 3 + config["repeats"] * len(config["concurrency"]) * (
        config["warmup_requests"] + len(config["prompts"])
    )
    if required > config["request_cap"]:
        raise ValueError("Schedule exceeds generation request cap")


async def run_condition(client, specs, writer, origin, concurrency, stop_at):
    """Workers pull only when free; there is no unbounded task queue or retry."""
    pending = iter(specs)
    outcomes = []

    async def worker():
        while time.perf_counter() + client.deadline_s < stop_at:
            spec = next(pending, None)
            if spec is None:
                break
            result = await client.generate(spec, origin=origin)
            writer.save(result)
            outcomes.append(result)
            if asyncio.current_task().cancelling():
                break

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*workers)
    finally:
        for worker_task in workers:
            if not worker_task.done():
                worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    return outcomes


async def run_experiment(
    base_url: str, config: dict, directory: Path, metadata: dict | None = None
) -> dict:
    validate(config)
    origin = time.perf_counter()
    stop_at = origin + config["run_deadline_s"] - config["flush_reserve_s"]
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "status": "incomplete",
        "config": config,
        "config_sha256": content_hash(config),
        "metadata": metadata or {},
        "schedule": [],
        "generation_requests": 0,
    }
    writer = RunWriter(directory, manifest)
    completed = False
    error = None
    sequence = 0

    def specs(phase, repeat, concurrency, prompts, *, stream=True, cancel=False):
        nonlocal sequence
        values = []
        for prompt in prompts:
            sequence += 1
            if sequence > config["request_cap"]:
                raise ValueError("Request cap reached")
            values.append(
                RequestSpec(
                    f"r{sequence:04}",
                    prompt["id"],
                    prompt["messages"],
                    phase,
                    repeat,
                    concurrency,
                    stream,
                    512 if cancel else config["max_tokens"],
                    cancel,
                    prompt.get("expected_text"),
                )
            )
        manifest["schedule"].extend(s.request_id for s in values)
        return values

    try:
        async with ChatClient(
            base_url,
            config["model"],
            deadline_s=config["request_deadline_s"],
            idle_timeout_s=config["idle_timeout_s"],
            max_connections=max(config["concurrency"]),
        ) as client:
            async with asyncio.timeout(config["run_deadline_s"] - config["flush_reserve_s"]):
                response = await client.http.get("models")
                response.raise_for_status()
                models = response.json()
                manifest["models_response"] = models
                if config["model"] not in [m.get("id") for m in models.get("data", [])]:
                    raise ValueError("Configured model missing from /v1/models")

                async def condition(items, concurrency):
                    results = await run_condition(
                        client, items, writer, origin, concurrency, stop_at
                    )
                    manifest["generation_requests"] += len(results)
                    if len(results) != len(items):
                        raise TimeoutError("Insufficient remaining time; partial run saved")
                    return results

                for stream in (False, True):
                    results = await condition(
                        specs("smoke", 0, 1, config["prompts"][:1], stream=stream), 1
                    )
                    result = results[0]
                    if result.status != "success" or not result.text.strip():
                        raise ValueError(f"Contract smoke failed: {result.status}")
                    if result.output_tokens is None or result.prompt_tokens is None:
                        raise ValueError("Contract smoke requires server-reported token usage")
                    if result.response_model != config["model"]:
                        raise ValueError("Response model does not match requested model")
                for repeat in range(config["repeats"]):
                    order = config["concurrency"][:: 1 if repeat % 2 == 0 else -1]
                    for concurrency in order:
                        warmups = [
                            config["prompts"][i % len(config["prompts"])]
                            for i in range(config["warmup_requests"])
                        ]
                        warmed = await condition(
                            specs("warmup", repeat, concurrency, warmups), concurrency
                        )
                        if any(r.status != "success" for r in warmed):
                            raise ValueError("Warmup failed; stop before measuring this condition")
                        await condition(
                            specs("measure", repeat, concurrency, config["prompts"]), concurrency
                        )
                probe = {
                    "id": "cancellation",
                    "messages": [
                        {
                            "role": "user",
                            "content": "List the integers from 1 to 200, separated by spaces.",
                        }
                    ],
                }
                await condition(specs("probe", 0, 1, [probe], cancel=True), 1)
                completed = True
    except (Exception, asyncio.CancelledError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        # Count persisted records even when cancellation interrupts a condition.
        writer.requests.flush()
        with (directory / "requests.jsonl").open() as saved:
            count = sum(1 for _ in saved)
        writer.close(
            status="completed" if completed else "incomplete",
            error=error,
            generation_requests=count,
            elapsed_s=time.perf_counter() - origin,
        )
    return writer.manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = asyncio.run(
        run_experiment(
            args.base_url,
            json.loads(args.config.read_text()),
            args.output,
            {"topology": "user-specified endpoint"},
        )
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "error": manifest["error"],
                "generation_requests": manifest["generation_requests"],
            },
            indent=2,
        )
    )
    if manifest["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
