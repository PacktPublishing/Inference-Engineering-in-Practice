"""Inspectable static and continuous batching simulations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Policy = Literal["static", "continuous"]


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """A request represented by its required decode iterations."""

    request_id: str
    decode_steps: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if self.decode_steps <= 0:
            raise ValueError("decode_steps must be positive")


@dataclass(frozen=True, slots=True)
class ScheduleResult:
    """Timeline and accounting produced by one scheduling policy."""

    policy: Policy
    batch_size: int
    timeline: tuple[tuple[str | None, ...], ...]
    started_at: tuple[tuple[str, int], ...]
    completed_at: tuple[tuple[str, int], ...]
    tokens_processed: int

    @property
    def total_steps(self) -> int:
        return len(self.timeline)

    @property
    def active_slot_steps(self) -> int:
        return sum(slot is not None for row in self.timeline for slot in row)

    @property
    def capacity_slot_steps(self) -> int:
        return self.total_steps * self.batch_size

    @property
    def idle_slot_steps(self) -> int:
        return self.capacity_slot_steps - self.active_slot_steps

    @property
    def utilization(self) -> float:
        if self.capacity_slot_steps == 0:
            return 0.0
        return self.active_slot_steps / self.capacity_slot_steps

    def start_times(self) -> dict[str, int]:
        return dict(self.started_at)

    def completion_times(self) -> dict[str, int]:
        return dict(self.completed_at)


def _validated_requests(requests: Sequence[RequestSpec], batch_size: int) -> None:
    if not requests:
        raise ValueError("requests cannot be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request IDs must be unique")


def schedule_static(
    requests: Sequence[RequestSpec],
    *,
    batch_size: int,
) -> ScheduleResult:
    """Process fixed groups without admitting replacements until a group ends."""
    _validated_requests(requests, batch_size)
    timeline: list[tuple[str | None, ...]] = []
    started_at: dict[str, int] = {}
    completed_at: dict[str, int] = {}

    for offset in range(0, len(requests), batch_size):
        batch = list(requests[offset : offset + batch_size])
        group_start = len(timeline)
        for request in batch:
            started_at[request.request_id] = group_start
            completed_at[request.request_id] = group_start + request.decode_steps

        group_steps = max(request.decode_steps for request in batch)
        for local_step in range(group_steps):
            slots = [
                request.request_id if local_step < request.decode_steps else None
                for request in batch
            ]
            slots.extend([None] * (batch_size - len(slots)))
            timeline.append(tuple(slots))

    return ScheduleResult(
        policy="static",
        batch_size=batch_size,
        timeline=tuple(timeline),
        started_at=tuple(started_at.items()),
        completed_at=tuple(completed_at.items()),
        tokens_processed=sum(request.decode_steps for request in requests),
    )


def schedule_continuous(
    requests: Sequence[RequestSpec],
    *,
    batch_size: int,
) -> ScheduleResult:
    """Admit a waiting request at the next iteration after a slot becomes free."""
    _validated_requests(requests, batch_size)
    timeline: list[tuple[str | None, ...]] = []
    started_at: dict[str, int] = {}
    completed_at: dict[str, int] = {}
    next_request = 0
    slots: list[tuple[RequestSpec, int] | None] = [None] * batch_size

    while next_request < len(requests) or any(slot is not None for slot in slots):
        current_step = len(timeline)
        for slot_index, slot in enumerate(slots):
            if slot is None and next_request < len(requests):
                request = requests[next_request]
                next_request += 1
                slots[slot_index] = (request, request.decode_steps)
                started_at[request.request_id] = current_step

        timeline.append(
            tuple(slot[0].request_id if slot is not None else None for slot in slots)
        )

        for slot_index, slot in enumerate(slots):
            if slot is None:
                continue
            request, remaining = slot
            remaining -= 1
            if remaining == 0:
                completed_at[request.request_id] = current_step + 1
                slots[slot_index] = None
            else:
                slots[slot_index] = (request, remaining)

    return ScheduleResult(
        policy="continuous",
        batch_size=batch_size,
        timeline=tuple(timeline),
        started_at=tuple(started_at.items()),
        completed_at=tuple(completed_at.items()),
        tokens_processed=sum(request.decode_steps for request in requests),
    )
