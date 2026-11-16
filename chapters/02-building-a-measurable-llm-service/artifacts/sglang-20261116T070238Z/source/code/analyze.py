"""Recompute descriptive measurements from saved terminal records, without a GPU."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evidence import validate_events, validate_run  # noqa: F401


def summarize(rows: list[dict], objective_s: float) -> list[dict]:
    seen = set()
    groups = defaultdict(list)
    for row in rows:
        if row["request_id"] in seen:
            raise ValueError("Duplicate request ID")
        seen.add(row["request_id"])
        if row["ended_s"] < row["started_s"] or row["status"] == "pending":
            raise ValueError("Invalid terminal record")
        if row["phase"] == "measure":
            groups[row["repeat"], row["concurrency"]].append(row)
    output = []
    for (repeat, concurrency), samples in sorted(groups.items()):
        duration = max(r["ended_s"] for r in samples) - min(r["started_s"] for r in samples)
        if duration <= 0:
            raise ValueError("Measurement window must be positive")
        success = [r for r in samples if r["status"] == "success"]
        counted = [
            r
            for r in success
            if r["output_tokens"] is not None and r["usage_source"] == "server_reported"
        ]
        useful = [
            r
            for r in success
            if r["output_check_passed"] is True and r["ended_s"] - r["started_s"] <= objective_s
        ]
        result = {
            "repeat": repeat,
            "concurrency": concurrency,
            "requests": len(samples),
            "successes": len(success),
            "failures": len(samples) - len(success),
            "window_s": duration,
            "failure_fraction": 1 - len(success) / len(samples),
            "requests_per_s": len(success) / duration,
            "counted_output_tokens_per_s": sum(r["output_tokens"] for r in counted) / duration
            if counted
            else None,
            "token_count_coverage": len(counted) / len(success) if success else None,
            "objective_fraction": len(useful) / len(samples),
            "objective_requests_per_s": len(useful) / duration,
        }
        distributions = {
            "completion_s": [r["ended_s"] - r["started_s"] for r in success],
            "first_content_s": [
                r["first_content_s"] for r in success if r["first_content_s"] is not None
            ],
            "output_tokens": [r["output_tokens"] for r in counted],
            "approx_output_interval_s": [
                (r["last_content_s"] - r["first_content_s"]) / (r["output_tokens"] - 1)
                for r in counted
                if r["output_tokens"] > 1
                and r["first_content_s"] is not None
                and r["last_content_s"] is not None
            ],
        }
        for name, values in distributions.items():
            result[f"{name}_n"] = len(values)
            for stat, fn in [("median", statistics.median), ("min", min), ("max", max)]:
                result[f"{name}_{stat}"] = fn(values) if values else None
        output.append(result)
    return output


def analyze(directory: Path, *, plots: bool = True, output: Path | None = None):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    validate_run(manifest, rows, events)
    summaries = summarize(rows, manifest["config"]["objective_latency_s"])
    if not summaries:
        raise ValueError("No measured requests; raw failure evidence retained")
    output = output or directory
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, field, label in zip(
            axes,
            ["first_content_s_median", "completion_s_median", "requests_per_s"],
            ["First content (s), median", "Completion (s), median", "Successful requests/s"],
            strict=True,
        ):
            for repeat in sorted({s["repeat"] for s in summaries}):
                selected = [s for s in summaries if s["repeat"] == repeat]
                axis.plot(
                    [s["concurrency"] for s in selected],
                    [s[field] if s[field] is not None else float("nan") for s in selected],
                    "o-",
                    label=f"Repeat {repeat + 1}",
                )
            axis.set(xlabel="Client concurrency", ylabel=label)
            axis.legend()
        label = manifest["metadata"].get("evidence_kind", "measured endpoint")
        figure.suptitle(f"{label} — {manifest['status']}; small descriptive samples")
        figure.tight_layout()
        figure.savefig(output / "measurements.png", dpi=160)
        plt.close(figure)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Write derived files separately from raw evidence"
    )
    args = parser.parse_args()
    print(
        json.dumps(analyze(args.directory, plots=not args.no_plots, output=args.output), indent=2)
    )
