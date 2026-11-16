"""Recalculate the book's comparison from its compact, measured-request CSV.

This checks arithmetic, not omitted event traces, warmups, or deployment controls.
Use compare.py for full evidence validation of new runs.
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from analyze import summarize
from compare import plot


def read_measurements(path: Path) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            for key in ("repeat", "concurrency", "prompt_tokens", "output_tokens"):
                row[key] = int(row[key]) if row[key] else None
            for key in ("started_s", "ended_s", "first_content_s", "last_content_s"):
                row[key] = float(row[key]) if row[key] else None
                if row[key] is not None and (not math.isfinite(row[key]) or row[key] < 0):
                    raise ValueError(f"Invalid {key}")
            if row["started_s"] is None or row["ended_s"] is None:
                raise ValueError("Missing condition timing")
            if row["output_check_passed"] not in ("True", "False", ""):
                raise ValueError("Invalid output check")
            row["output_check_passed"] = {"True": True, "False": False, "": None}[
                row["output_check_passed"]
            ]
            if row["phase"] != "measure":
                raise ValueError("Expected measured requests only")
            groups[row["engine"]].append(row)
    if not groups:
        raise ValueError("No measurements")
    if any(len({r["run_id"] for r in rows}) != 1 for rows in groups.values()):
        raise ValueError("Do not pool different runs of one engine")
    return dict(groups)


def reproduce(path: Path, output: Path) -> list[dict]:
    results = [
        {"engine": engine, **summary}
        for engine, rows in read_measurements(path).items()
        for summary in summarize(rows, objective_s=5)
    ]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    # Hardware and objective belong to this book example, not arbitrary input runs.
    plot(results, output, gpu="L40S:1")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reproduce(args.measurements, args.output)
    print(f"Recalculated book example in {args.output}; full run evidence was not checked")
