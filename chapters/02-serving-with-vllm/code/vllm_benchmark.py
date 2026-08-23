"""Transparent request-level accounting for Chapter 2 serving benchmarks."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """Client-observed timing and token counts for one request."""

    request_id: str
    started_at: float
    completed_at: float
    first_content_at: float | None
    prompt_tokens: int
    output_tokens: int
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        if self.first_content_at is not None and not (
            self.started_at <= self.first_content_at <= self.completed_at
        ):
            raise ValueError("first_content_at must fall within the request interval")
        if self.prompt_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts cannot be negative")
        if self.error is None and self.first_content_at is None:
            raise ValueError("a successful request must contain generated content")

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def e2e_seconds(self) -> float:
        return self.completed_at - self.started_at

    @property
    def ttft_seconds(self) -> float | None:
        if self.first_content_at is None:
            return None
        return self.first_content_at - self.started_at

    @property
    def tpot_seconds(self) -> float | None:
        """Client-side approximation; streamed chunks are not assumed to be tokens."""
        ttft = self.ttft_seconds
        if ttft is None or self.output_tokens == 0:
            return None
        return (self.e2e_seconds - ttft) / max(self.output_tokens - 1, 1)


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """A small explicit latency distribution summary."""

    count: int
    minimum: float
    mean: float
    p50: float
    p95: float
    p99: float
    maximum: float


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Aggregate benchmark values derived only from raw request records."""

    request_count: int
    success_count: int
    error_count: int
    elapsed_seconds: float
    request_throughput: float
    input_token_throughput: float
    output_token_throughput: float
    ttft: DistributionSummary
    tpot: DistributionSummary
    e2e: DistributionSummary


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""
    if not values:
        raise ValueError("values cannot be empty")
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile_value / 100
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def summarize_distribution(values: Sequence[float]) -> DistributionSummary:
    """Summarize values without discarding their sample count."""
    if not values:
        raise ValueError("values cannot be empty")
    normalized = [float(value) for value in values]
    return DistributionSummary(
        count=len(normalized),
        minimum=min(normalized),
        mean=statistics.fmean(normalized),
        p50=percentile(normalized, 50),
        p95=percentile(normalized, 95),
        p99=percentile(normalized, 99),
        maximum=max(normalized),
    )


def summarize_requests(
    records: Iterable[RequestRecord],
    *,
    benchmark_started_at: float,
    benchmark_completed_at: float,
) -> BenchmarkSummary:
    """Aggregate successful requests while retaining failures in request counts."""
    if benchmark_completed_at <= benchmark_started_at:
        raise ValueError("benchmark duration must be positive")
    collected = tuple(records)
    if not collected:
        raise ValueError("records cannot be empty")

    successful = tuple(record for record in collected if record.succeeded)
    if not successful:
        raise ValueError("at least one request must succeed")
    if any(
        record.started_at < benchmark_started_at
        or record.completed_at > benchmark_completed_at
        for record in collected
    ):
        raise ValueError("all request intervals must fall within the benchmark interval")

    ttft_values = [
        record.ttft_seconds
        for record in successful
        if record.ttft_seconds is not None
    ]
    tpot_values = [
        record.tpot_seconds
        for record in successful
        if record.tpot_seconds is not None
    ]
    e2e_values = [record.e2e_seconds for record in successful]
    if len(ttft_values) != len(successful) or len(tpot_values) != len(successful):
        raise ValueError("successful records must contain TTFT and TPOT values")

    elapsed_seconds = benchmark_completed_at - benchmark_started_at
    input_tokens = sum(record.prompt_tokens for record in successful)
    output_tokens = sum(record.output_tokens for record in successful)
    return BenchmarkSummary(
        request_count=len(collected),
        success_count=len(successful),
        error_count=len(collected) - len(successful),
        elapsed_seconds=elapsed_seconds,
        request_throughput=len(successful) / elapsed_seconds,
        input_token_throughput=input_tokens / elapsed_seconds,
        output_token_throughput=output_tokens / elapsed_seconds,
        ttft=summarize_distribution(ttft_values),
        tpot=summarize_distribution(tpot_values),
        e2e=summarize_distribution(e2e_values),
    )


def estimated_cost_per_million_tokens(
    *,
    gpu_seconds: float,
    gpu_price_per_second: float,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Normalize measured GPU time to one million processed tokens."""
    if gpu_seconds < 0 or gpu_price_per_second < 0:
        raise ValueError("GPU time and price cannot be negative")
    total_tokens = input_tokens + output_tokens
    if total_tokens <= 0:
        raise ValueError("total token count must be positive")
    return gpu_seconds * gpu_price_per_second * 1_000_000 / total_tokens
