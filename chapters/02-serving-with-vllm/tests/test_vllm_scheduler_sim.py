import pytest
from vllm_scheduler_sim import (
    RequestSpec,
    schedule_continuous,
    schedule_static,
)


def test_continuous_batching_refills_slots_at_iteration_boundaries() -> None:
    requests = [
        RequestSpec("A", 2),
        RequestSpec("B", 4),
        RequestSpec("C", 1),
        RequestSpec("D", 3),
    ]

    static = schedule_static(requests, batch_size=2)
    continuous = schedule_continuous(requests, batch_size=2)

    assert static.timeline == (
        ("A", "B"),
        ("A", "B"),
        (None, "B"),
        (None, "B"),
        ("C", "D"),
        (None, "D"),
        (None, "D"),
    )
    assert continuous.timeline == (
        ("A", "B"),
        ("A", "B"),
        ("C", "B"),
        ("D", "B"),
        ("D", None),
        ("D", None),
    )
    assert continuous.start_times()["C"] == 2
    assert continuous.start_times()["D"] == 3


def test_both_policies_process_the_same_token_work() -> None:
    requests = [RequestSpec("short", 1), RequestSpec("long", 5), RequestSpec("new", 2)]

    static = schedule_static(requests, batch_size=2)
    continuous = schedule_continuous(requests, batch_size=2)

    assert static.tokens_processed == continuous.tokens_processed == 8
    assert static.active_slot_steps == continuous.active_slot_steps == 8
    assert continuous.total_steps < static.total_steps
    assert continuous.utilization > static.utilization


def test_scheduler_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        RequestSpec("request", 0)

    duplicate = [RequestSpec("same", 1), RequestSpec("same", 2)]
    with pytest.raises(ValueError, match="unique"):
        schedule_continuous(duplicate, batch_size=2)

    with pytest.raises(ValueError, match="batch_size"):
        schedule_static([RequestSpec("one", 1)], batch_size=0)
