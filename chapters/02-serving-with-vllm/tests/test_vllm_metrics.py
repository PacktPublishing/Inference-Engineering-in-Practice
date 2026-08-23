import math

import pytest
from vllm_metrics import (
    MetricSample,
    counter_deltas,
    index_samples,
    parse_prometheus_text,
    select_samples,
)


def test_parse_prometheus_samples_with_labels_and_special_values() -> None:
    text = r'''
# HELP vllm:num_requests_running Number of running requests.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="chapter-2",worker="gpu-0"} 3
vllm:request_success_total{finish_reason="stop",note="line\n\"quoted\""} 12 17
vllm:cache_usage +Inf
'''

    samples = parse_prometheus_text(text)

    assert samples[0] == MetricSample(
        name="vllm:num_requests_running",
        labels=(("model_name", "chapter-2"), ("worker", "gpu-0")),
        value=3.0,
    )
    assert samples[1].label_dict()["note"] == 'line\n"quoted"'
    assert samples[1].timestamp == 17.0
    assert math.isinf(samples[2].value)


def test_select_samples_matches_a_label_subset() -> None:
    samples = parse_prometheus_text(
        """
metric_total{model="a",status="ok"} 2
metric_total{model="a",status="error"} 1
metric_total{model="b",status="ok"} 5
"""
    )

    selected = select_samples(samples, "metric_total", labels={"model": "a"})

    assert [sample.value for sample in selected] == [2.0, 1.0]


def test_counter_deltas_require_matching_monotonic_samples() -> None:
    before = parse_prometheus_text('requests_total{model="a"} 10')
    after = parse_prometheus_text('requests_total{model="a"} 14')

    deltas = counter_deltas(before, after, names={"requests_total"})

    assert next(iter(deltas.values())) == 4

    reset = parse_prometheus_text('requests_total{model="a"} 2')
    with pytest.raises(ValueError, match="reset"):
        counter_deltas(after, reset)

    new_series = parse_prometheus_text(
        'requests_total{model="a"} 14\nrequests_total{model="b"} 3'
    )
    with pytest.raises(ValueError, match="no baseline"):
        counter_deltas(before, new_series)
    allowed = counter_deltas(before, new_series, allow_new_series=True)
    assert allowed[("requests_total", (("model", "b"),))] == 3


def test_invalid_or_duplicate_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid Prometheus"):
        parse_prometheus_text("not a metric line")

    duplicate = parse_prometheus_text("metric 1\nmetric 2")
    with pytest.raises(ValueError, match="duplicate metric"):
        index_samples(duplicate)

    with pytest.raises(ValueError, match="duplicate label"):
        parse_prometheus_text('metric{model="a",model="b"} 1')
