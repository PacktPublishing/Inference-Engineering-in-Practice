"""Verify a preserved service bundle and write derived analysis elsewhere."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from analyze import analyze
from compare import read_run
from records import write_json
from service import validate_metrics


def verify(bundle: Path, output: Path) -> dict:
    if output.resolve() == bundle.resolve() or bundle.resolve() in output.resolve().parents:
        raise ValueError("Write verification outside the raw bundle")
    sums = json.loads((bundle / "SHA256SUMS.json").read_text())
    for name, expected in sums.items():
        path = (bundle / name).resolve()
        if bundle.resolve() not in path.parents:
            raise ValueError("Checksum path escapes the bundle")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    outcome = json.loads((bundle / "outcome.json").read_text())
    if outcome["status"] != "completed":
        raise ValueError("Service outcome is incomplete")
    manifest, rows = read_run(bundle / "run")
    source = manifest["metadata"]["source"]
    for name, expected in source["files"].items():
        path = (bundle / "source" / name).resolve()
        if (bundle / "source").resolve() not in path.parents:
            raise ValueError("Source path escapes the snapshot")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Executed source hash mismatch: {name}")
    if (
        hashlib.sha256((bundle / "source/experiment.json").read_bytes()).hexdigest()
        != source["config_sha256"]
    ):
        raise ValueError("Executed configuration hash mismatch")
    metrics = {
        phase: validate_metrics(
            (bundle / f"metrics-{phase}.txt").read_text(), manifest["metadata"]["engine"]
        )
        for phase in ("before", "after")
    }
    output.mkdir(parents=True, exist_ok=False)
    summaries = analyze(bundle / "run", output=output)
    report = {
        "status": "validated offline",
        "bundle": bundle.name,
        "checked_hashes": len(sums),
        "source_hashes": "matched",
        "event_and_schedule_checks": "passed",
        "strict_output_checks": "passed",
        "terminal_records": len(rows),
        "metric_gauges": metrics,
        "summaries": summaries,
        "limits": [
            "Hashes detect disagreement, not independently authenticated provenance",
            "Historical snapshots without HTTP observation records cannot prove HTTP status",
            "No live cloud state or per-request engine cancellation checked",
        ],
    }
    write_json(output / "verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.bundle, args.output)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("summaries", "metric_gauges")}, indent=2
        )
    )
