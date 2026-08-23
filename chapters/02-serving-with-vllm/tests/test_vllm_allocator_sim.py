import pytest
from vllm_allocator_sim import (
    CacheRequest,
    allocate_contiguous,
    allocate_paged,
)


def test_paged_allocation_admits_more_mixed_requests() -> None:
    requests = [
        CacheRequest("A", used_tokens=100, reserved_tokens=1024),
        CacheRequest("B", used_tokens=200, reserved_tokens=1024),
        CacheRequest("C", used_tokens=300, reserved_tokens=1024),
        CacheRequest("D", used_tokens=400, reserved_tokens=1024),
    ]

    contiguous = allocate_contiguous(requests, total_slots=2048)
    paged = allocate_paged(requests, total_slots=2048, block_size=16)

    assert contiguous.admitted_requests == 2
    assert paged.admitted_requests == 4
    assert contiguous.used_slots == 300
    assert paged.used_slots == 1000
    assert paged.wasted_slots == 24
    assert paged.allocation_utilization > contiguous.allocation_utilization


def test_paged_tail_waste_is_bounded_per_admitted_request() -> None:
    requests = [
        CacheRequest("one", used_tokens=1, reserved_tokens=128),
        CacheRequest("seventeen", used_tokens=17, reserved_tokens=128),
        CacheRequest("exact", used_tokens=32, reserved_tokens=128),
    ]

    result = allocate_paged(requests, total_slots=256, block_size=16)

    assert all(
        record.wasted_slots < 16
        for record in result.records
        if record.admitted
    )
    assert result.capacity_utilization == pytest.approx(50 / 256)


def test_allocator_rejects_invalid_capacity_and_requests() -> None:
    with pytest.raises(ValueError, match="reserved_tokens"):
        CacheRequest("invalid", used_tokens=10, reserved_tokens=9)

    requests = [CacheRequest("one", used_tokens=10, reserved_tokens=16)]
    with pytest.raises(ValueError, match="divisible"):
        allocate_paged(requests, total_slots=33, block_size=16)

    duplicates = [
        CacheRequest("same", used_tokens=1, reserved_tokens=16),
        CacheRequest("same", used_tokens=2, reserved_tokens=16),
    ]
    with pytest.raises(ValueError, match="unique"):
        allocate_contiguous(duplicates, total_slots=32)
