"""Generate deterministic Chapter 2 simulation artifacts on a local CPU."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "inference-engineering-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from vllm_allocator_sim import (  # noqa: E402
    AllocationSummary,
    CacheRequest,
    allocate_contiguous,
    allocate_paged,
)
from vllm_scheduler_sim import (  # noqa: E402
    RequestSpec,
    ScheduleResult,
    schedule_continuous,
    schedule_static,
)

CHAPTER_DIR = Path(__file__).parents[1]
DEFAULT_ARTIFACT_DIR = CHAPTER_DIR / "artifacts"

SCHEDULER_REQUESTS = (
    RequestSpec("A", 8),
    RequestSpec("B", 2),
    RequestSpec("C", 2),
    RequestSpec("D", 8),
    RequestSpec("E", 2),
    RequestSpec("F", 2),
)
SCHEDULER_BATCH_SIZE = 3

ALLOCATOR_REQUESTS = (
    CacheRequest("A", used_tokens=100, reserved_tokens=1024),
    CacheRequest("B", used_tokens=200, reserved_tokens=1024),
    CacheRequest("C", used_tokens=300, reserved_tokens=1024),
    CacheRequest("D", used_tokens=400, reserved_tokens=1024),
)
ALLOCATOR_TOTAL_SLOTS = 2048
ALLOCATOR_BLOCK_SIZE = 16


def _schedule_document(result: ScheduleResult) -> dict:
    return {
        "policy": result.policy,
        "batch_size": result.batch_size,
        "total_steps": result.total_steps,
        "tokens_processed": result.tokens_processed,
        "active_slot_steps": result.active_slot_steps,
        "idle_slot_steps": result.idle_slot_steps,
        "capacity_slot_steps": result.capacity_slot_steps,
        "utilization": result.utilization,
        "started_at": result.start_times(),
        "completed_at": result.completion_times(),
        "timeline": [list(row) for row in result.timeline],
    }


def _allocation_document(
    result: AllocationSummary,
    requests: Sequence[CacheRequest],
) -> dict:
    request_by_id = {request.request_id: request for request in requests}
    return {
        "policy": result.policy,
        "total_slots": result.total_slots,
        "block_size": result.block_size,
        "admitted_requests": result.admitted_requests,
        "rejected_requests": result.rejected_requests,
        "used_slots": result.used_slots,
        "allocated_slots": result.allocated_slots,
        "wasted_slots": result.wasted_slots,
        "free_slots": result.free_slots,
        "allocation_utilization": result.allocation_utilization,
        "capacity_utilization": result.capacity_utilization,
        "records": [
            {
                "request_id": record.request_id,
                "requested_used_tokens": request_by_id[record.request_id].used_tokens,
                "requested_reserved_tokens": request_by_id[
                    record.request_id
                ].reserved_tokens,
                "admitted": record.admitted,
                "used_slots": record.used_slots,
                "allocated_slots": record.allocated_slots,
                "wasted_slots": record.wasted_slots,
            }
            for record in result.records
        ],
    }


def _write_scheduler_csv(
    destination: Path,
    results: Sequence[ScheduleResult],
) -> None:
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("policy", "step", "slot", "request_id"),
        )
        writer.writeheader()
        for result in results:
            for step, row in enumerate(result.timeline):
                for slot, request_id in enumerate(row):
                    writer.writerow(
                        {
                            "policy": result.policy,
                            "step": step,
                            "slot": slot,
                            "request_id": "" if request_id is None else request_id,
                        }
                    )


def _write_allocator_csv(
    destination: Path,
    results: Sequence[AllocationSummary],
) -> None:
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "policy",
                "request_id",
                "admitted",
                "used_slots",
                "allocated_slots",
                "wasted_slots",
            ),
        )
        writer.writeheader()
        for result in results:
            for record in result.records:
                writer.writerow(
                    {
                        "policy": result.policy,
                        "request_id": record.request_id,
                        "admitted": record.admitted,
                        "used_slots": record.used_slots,
                        "allocated_slots": record.allocated_slots,
                        "wasted_slots": record.wasted_slots,
                    }
                )


def _plot_scheduler(
    destination: Path,
    results: Sequence[ScheduleResult],
) -> None:
    request_ids = sorted(
        {
            request_id
            for result in results
            for row in result.timeline
            for request_id in row
            if request_id is not None
        }
    )
    palette = plt.get_cmap("tab10")
    colors = {
        request_id: palette(index % 10)
        for index, request_id in enumerate(request_ids)
    }

    figure, axes = plt.subplots(
        len(results),
        1,
        figsize=(10, 4.8),
        constrained_layout=True,
    )
    for axis, result in zip(axes, results, strict=True):
        for step, row in enumerate(result.timeline):
            for slot, request_id in enumerate(row):
                if request_id is None:
                    continue
                axis.add_patch(
                    Rectangle(
                        (step, slot),
                        1,
                        1,
                        facecolor=colors[request_id],
                        edgecolor="white",
                    )
                )
                axis.text(
                    step + 0.5,
                    slot + 0.5,
                    request_id,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
        axis.set(
            title=(
                f"{result.policy.title()} batching — "
                f"{result.utilization:.0%} slot utilization"
            ),
            xlim=(0, result.total_steps),
            ylim=(result.batch_size, 0),
            ylabel="Batch slot",
        )
        axis.set_xticks(range(result.total_steps + 1))
        axis.set_yticks(
            [index + 0.5 for index in range(result.batch_size)],
            labels=[str(index) for index in range(result.batch_size)],
        )
        axis.grid(axis="x", color="#dddddd", linewidth=0.6)
    axes[-1].set_xlabel("Decode iteration")
    figure.legend(
        handles=[
            Patch(facecolor=colors[request_id], label=f"Request {request_id}")
            for request_id in request_ids
        ],
        loc="outside upper right",
        frameon=False,
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _plot_allocator(
    destination: Path,
    results: Sequence[AllocationSummary],
) -> None:
    labels = [result.policy.title() for result in results]
    used = [result.used_slots for result in results]
    wasted = [result.wasted_slots for result in results]
    free = [result.free_slots for result in results]

    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.bar(labels, used, label="Used token slots", color="#3b82f6")
    axis.bar(labels, wasted, bottom=used, label="Allocated but unused", color="#f59e0b")
    axis.bar(
        labels,
        free,
        bottom=[used_value + waste for used_value, waste in zip(used, wasted, strict=True)],
        label="Unallocated",
        color="#d1d5db",
    )
    for index, result in enumerate(results):
        axis.text(
            index,
            result.total_slots + result.total_slots * 0.02,
            f"{result.admitted_requests} admitted",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    axis.set(
        title="Same KV capacity, different allocation policies",
        ylabel="KV token slots",
        ylim=(0, ALLOCATOR_TOTAL_SLOTS * 1.12),
    )
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncols=3,
    )
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def write_local_artifacts(output_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> tuple[Path, ...]:
    """Generate the deterministic scheduler and allocator artifact set."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    static = schedule_static(SCHEDULER_REQUESTS, batch_size=SCHEDULER_BATCH_SIZE)
    continuous = schedule_continuous(
        SCHEDULER_REQUESTS,
        batch_size=SCHEDULER_BATCH_SIZE,
    )
    contiguous = allocate_contiguous(
        ALLOCATOR_REQUESTS,
        total_slots=ALLOCATOR_TOTAL_SLOTS,
    )
    paged = allocate_paged(
        ALLOCATOR_REQUESTS,
        total_slots=ALLOCATOR_TOTAL_SLOTS,
        block_size=ALLOCATOR_BLOCK_SIZE,
    )

    scheduler_json = destination / "scheduler-simulation.json"
    scheduler_csv = destination / "scheduler-timeline.csv"
    scheduler_plot = destination / "scheduler-simulation.png"
    allocator_json = destination / "allocator-simulation.json"
    allocator_csv = destination / "allocator-records.csv"
    allocator_plot = destination / "allocator-simulation.png"

    scheduler_json.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "request_id": request.request_id,
                        "decode_steps": request.decode_steps,
                    }
                    for request in SCHEDULER_REQUESTS
                ],
                "static": _schedule_document(static),
                "continuous": _schedule_document(continuous),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    allocator_json.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "request_id": request.request_id,
                        "used_tokens": request.used_tokens,
                        "reserved_tokens": request.reserved_tokens,
                    }
                    for request in ALLOCATOR_REQUESTS
                ],
                "contiguous": _allocation_document(contiguous, ALLOCATOR_REQUESTS),
                "paged": _allocation_document(paged, ALLOCATOR_REQUESTS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_scheduler_csv(scheduler_csv, (static, continuous))
    _write_allocator_csv(allocator_csv, (contiguous, paged))
    _plot_scheduler(scheduler_plot, (static, continuous))
    _plot_allocator(allocator_plot, (contiguous, paged))

    return (
        scheduler_json,
        scheduler_csv,
        scheduler_plot,
        allocator_json,
        allocator_csv,
        allocator_plot,
    )


if __name__ == "__main__":
    for artifact_path in write_local_artifacts():
        print(artifact_path)
