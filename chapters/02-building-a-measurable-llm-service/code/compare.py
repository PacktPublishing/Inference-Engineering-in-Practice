"""Compare two saved serving setups after checking their declared experiment controls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze import summarize
from evidence import validate_run
from prepare_model import validate_preparation
from records import write_json


def read_run(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    if manifest["status"] != "completed":
        raise ValueError("Do not silently compare an incomplete run as a completed workload")
    validate_run(manifest, rows, events)
    # Legacy bundles predate prompt hashes; submitted messages are checked above.
    validate_preparation(
        manifest["config"], manifest["metadata"]["model_preparation"], require_prompt_hash=False
    )
    if manifest["generation_requests"] != len(rows):
        raise ValueError("Manifest and terminal record counts disagree")
    return manifest, rows


def compare(first: Path, second: Path, output: Path):
    runs = [read_run(first), read_run(second)]
    left, right = runs[0][0], runs[1][0]
    controls = [
        "model",
        "model_revision",
        "prompts",
        "max_tokens",
        "repeats",
        "concurrency",
        "warmup_requests",
        "objective_latency_s",
        "request_deadline_s",
        "idle_timeout_s",
    ]
    for key in controls:
        if left["config"][key] != right["config"][key]:
            raise ValueError(f"Experiment control differs: {key}")
    for key in ["resources", "topology", "execution_mode"]:
        if left["metadata"][key] != right["metadata"][key]:
            raise ValueError(f"Serving control differs: {key}")
    if (
        left["metadata"]["model_preparation"]["chat_template_sha256"]
        != right["metadata"]["model_preparation"]["chat_template_sha256"]
    ):
        raise ValueError("Chat template hashes differ")
    left_prompts = left["metadata"]["model_preparation"]["prompt_tokens"]
    for manifest, rows in runs:
        measured = [r for r in rows if r["phase"] == "measure"]
        if len(measured) != (
            len(manifest["config"]["prompts"])
            * manifest["config"]["repeats"]
            * len(manifest["config"]["concurrency"])
        ):
            raise ValueError("Measured request count does not match the declared workload")
        if any(
            r["prompt_tokens"] is not None and r["prompt_tokens"] != left_prompts[r["prompt_id"]]
            for r in measured
        ):
            raise ValueError("Actual formatted prompt counts differ")
    results = []
    for manifest, rows in runs:
        results.extend(
            {"engine": manifest["metadata"]["engine"], **row}
            for row in summarize(rows, manifest["config"]["objective_latency_s"])
        )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    write_json(
        output / "comparison.json",
        {
            "runs": [
                {
                    "engine": m["metadata"]["engine"],
                    "created_utc": m["created_utc"],
                    "config_sha256": m["config_sha256"],
                    "versions": m["metadata"]["versions"],
                }
                for m, _ in runs
            ],
            "checked_controls": controls
            + [
                "resources",
                "topology",
                "execution_mode",
                "chat_template_sha256",
                "actual_prompt_tokens",
            ],
            "limits": [
                "Complete serving setups including different API frontends and dependencies",
                "Engine memory/scheduler controls have engine-specific semantics",
                "Small sequential repeats; no stable tail or production capacity claim",
                "Actual output lengths can differ despite the same model and generation policy",
            ],
            "summaries": results,
        },
    )
    plot(results, output)
    return results


def plot(results, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    engines = sorted({r["engine"] for r in results})
    for axis, metric, title, scale in zip(
        axes,
        ["first_content_s_median", "completion_s_median", "counted_output_tokens_per_s"],
        ["Median first content (ms)", "Median completion (ms)", "Counted output tokens/s"],
        [1000, 1000, 1],
        strict=True,
    ):
        for index, engine in enumerate(engines):
            for repeat in sorted({r["repeat"] for r in results}):
                rows = [r for r in results if r["engine"] == engine and r["repeat"] == repeat]
                axis.plot(
                    [r["concurrency"] for r in rows],
                    [r[metric] * scale if r[metric] is not None else float("nan") for r in rows],
                    marker=["o", "s", "^"][repeat % 3],
                    color=f"C{index}",
                    alpha=0.75,
                    label=engine if repeat == 0 else None,
                )
        axis.set(xlabel="Client concurrency", ylabel=title, xticks=[1, 4])
        axis.legend()
    figure.suptitle("Qwen 1.5B · L4 · loopback API · 3 repeats × 24 requests per condition")
    figure.tight_layout()
    figure.savefig(output / "engine-comparison.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.first, args.second, args.output)
    print(f"Validated comparison written to {args.output}")
