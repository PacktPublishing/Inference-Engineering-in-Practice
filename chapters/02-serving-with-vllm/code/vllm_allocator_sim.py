"""Small KV-capacity model for contiguous reservation and paged allocation.

This simulator isolates over-reservation and block-tail waste. It is not an
implementation of vLLM's KV-cache manager or attention kernel.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

AllocationPolicy = Literal["contiguous", "paged"]


@dataclass(frozen=True, slots=True)
class CacheRequest:
    """Token slots used now and reserved by a contiguous baseline."""

    request_id: str
    used_tokens: int
    reserved_tokens: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if self.used_tokens <= 0:
            raise ValueError("used_tokens must be positive")
        if self.reserved_tokens < self.used_tokens:
            raise ValueError("reserved_tokens cannot be smaller than used_tokens")


@dataclass(frozen=True, slots=True)
class AllocationRecord:
    """One admitted or rejected request."""

    request_id: str
    admitted: bool
    used_slots: int
    allocated_slots: int

    @property
    def wasted_slots(self) -> int:
        return self.allocated_slots - self.used_slots


@dataclass(frozen=True, slots=True)
class AllocationSummary:
    """Aggregate capacity accounting for one allocation policy."""

    policy: AllocationPolicy
    total_slots: int
    block_size: int | None
    records: tuple[AllocationRecord, ...]

    @property
    def admitted_requests(self) -> int:
        return sum(record.admitted for record in self.records)

    @property
    def rejected_requests(self) -> int:
        return len(self.records) - self.admitted_requests

    @property
    def used_slots(self) -> int:
        return sum(record.used_slots for record in self.records if record.admitted)

    @property
    def allocated_slots(self) -> int:
        return sum(record.allocated_slots for record in self.records if record.admitted)

    @property
    def wasted_slots(self) -> int:
        return self.allocated_slots - self.used_slots

    @property
    def free_slots(self) -> int:
        return self.total_slots - self.allocated_slots

    @property
    def allocation_utilization(self) -> float:
        if self.allocated_slots == 0:
            return 0.0
        return self.used_slots / self.allocated_slots

    @property
    def capacity_utilization(self) -> float:
        return self.used_slots / self.total_slots


def _validate(requests: Sequence[CacheRequest], total_slots: int) -> None:
    if not requests:
        raise ValueError("requests cannot be empty")
    if total_slots <= 0:
        raise ValueError("total_slots must be positive")
    request_ids = [request.request_id for request in requests]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request IDs must be unique")


def allocate_contiguous(
    requests: Sequence[CacheRequest],
    *,
    total_slots: int,
) -> AllocationSummary:
    """Reserve each admitted request's declared maximum token capacity."""
    _validate(requests, total_slots)
    available = total_slots
    records: list[AllocationRecord] = []

    for request in requests:
        admitted = request.reserved_tokens <= available
        allocated = request.reserved_tokens if admitted else 0
        used = request.used_tokens if admitted else 0
        records.append(
            AllocationRecord(
                request_id=request.request_id,
                admitted=admitted,
                used_slots=used,
                allocated_slots=allocated,
            )
        )
        available -= allocated

    return AllocationSummary(
        policy="contiguous",
        total_slots=total_slots,
        block_size=None,
        records=tuple(records),
    )


def allocate_paged(
    requests: Sequence[CacheRequest],
    *,
    total_slots: int,
    block_size: int,
) -> AllocationSummary:
    """Allocate only enough fixed-size blocks for tokens currently in use."""
    _validate(requests, total_slots)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if total_slots % block_size:
        raise ValueError("total_slots must be divisible by block_size")

    available = total_slots
    records: list[AllocationRecord] = []
    for request in requests:
        required = math.ceil(request.used_tokens / block_size) * block_size
        admitted = required <= available
        allocated = required if admitted else 0
        used = request.used_tokens if admitted else 0
        records.append(
            AllocationRecord(
                request_id=request.request_id,
                admitted=admitted,
                used_slots=used,
                allocated_slots=allocated,
            )
        )
        available -= allocated

    return AllocationSummary(
        policy="paged",
        total_slots=total_slots,
        block_size=block_size,
        records=tuple(records),
    )
