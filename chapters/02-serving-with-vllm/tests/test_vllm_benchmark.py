import pytest
from vllm_benchmark import (
    RequestRecord,
    estimated_cost_per_million_tokens,
    percentile,
    summarize_requests,
)


def test_percentile_uses_linear_interpolation() -> None:
    values = [0.0, 10.0, 20.0, 30.0]

    assert percentile(values, 0) == 0
    assert percentile(values, 50) == 15
    assert percentile(values, 95) == pytest.approx(28.5)
    assert percentile(values, 100) == 30


def test_request_record_exposes_ttft_e2e_and_tpot() -> None:
    record = RequestRecord(
        request_id="request-1",
        started_at=1.0,
        first_content_at=1.2,
        completed_at=2.0,
        prompt_tokens=10,
        output_tokens=3,
    )

    assert record.ttft_seconds == pytest.approx(0.2)
    assert record.e2e_seconds == pytest.approx(1.0)
    assert record.tpot_seconds == pytest.approx(0.4)


def test_summary_retains_errors_but_aggregates_successful_requests() -> None:
    records = [
        RequestRecord(
            request_id="ok-1",
            started_at=0.0,
            first_content_at=0.2,
            completed_at=1.0,
            prompt_tokens=10,
            output_tokens=5,
        ),
        RequestRecord(
            request_id="ok-2",
            started_at=0.5,
            first_content_at=0.8,
            completed_at=1.5,
            prompt_tokens=20,
            output_tokens=10,
        ),
        RequestRecord(
            request_id="error-1",
            started_at=1.0,
            first_content_at=None,
            completed_at=1.2,
            prompt_tokens=5,
            output_tokens=0,
            error="timeout",
        ),
    ]

    summary = summarize_requests(
        records,
        benchmark_started_at=0.0,
        benchmark_completed_at=2.0,
    )

    assert summary.request_count == 3
    assert summary.success_count == 2
    assert summary.error_count == 1
    assert summary.request_throughput == 1
    assert summary.input_token_throughput == 15
    assert summary.output_token_throughput == 7.5
    assert summary.ttft.count == 2
    assert summary.e2e.p50 == 1


def test_invalid_success_and_intervals_are_rejected() -> None:
    with pytest.raises(ValueError, match="generated content"):
        RequestRecord(
            request_id="missing-content",
            started_at=0,
            first_content_at=None,
            completed_at=1,
            prompt_tokens=1,
            output_tokens=0,
        )

    record = RequestRecord(
        request_id="outside",
        started_at=0,
        first_content_at=0.1,
        completed_at=2,
        prompt_tokens=1,
        output_tokens=1,
    )
    with pytest.raises(ValueError, match="benchmark interval"):
        summarize_requests(
            [record],
            benchmark_started_at=0,
            benchmark_completed_at=1,
        )


def test_cost_is_normalized_by_processed_tokens() -> None:
    cost = estimated_cost_per_million_tokens(
        gpu_seconds=100,
        gpu_price_per_second=0.001,
        input_tokens=60_000,
        output_tokens=40_000,
    )

    assert cost == pytest.approx(1.0)
