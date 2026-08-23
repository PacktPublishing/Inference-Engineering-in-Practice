"""Small Prometheus exposition parser for pinned Chapter 2 metric snapshots."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

_METRIC_NAME = r"[a-zA-Z_:][a-zA-Z0-9_:]*"
_NUMBER = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)"
_SAMPLE_PATTERN = re.compile(
    rf"^(?P<name>{_METRIC_NAME})(?:\{{(?P<labels>.*)\}})?[ \t]+"
    rf"(?P<value>{_NUMBER})(?:[ \t]+(?P<timestamp>[-+]?\d+(?:\.\d+)?))?$"
)
_LABEL_PATTERN = re.compile(
    r'(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:\\.|[^"\\])*)"(?:,|$)'
)


@dataclass(frozen=True, slots=True)
class MetricSample:
    """One labeled sample from the Prometheus text exposition format."""

    name: str
    labels: tuple[tuple[str, str], ...]
    value: float
    timestamp: float | None = None

    @property
    def key(self) -> tuple[str, tuple[tuple[str, str], ...]]:
        return self.name, self.labels

    def label_dict(self) -> dict[str, str]:
        return dict(self.labels)


def _unescape_label(value: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "\\":
            output.append(character)
            index += 1
            continue
        if index + 1 >= len(value):
            raise ValueError("label value ends with an incomplete escape")
        escaped = value[index + 1]
        replacements = {"\\": "\\", '"': '"', "n": "\n"}
        if escaped not in replacements:
            raise ValueError(f"unsupported label escape: \\{escaped}")
        output.append(replacements[escaped])
        index += 2
    return "".join(output)


def _parse_labels(raw_labels: str | None, *, line_number: int) -> tuple[tuple[str, str], ...]:
    if raw_labels is None or not raw_labels.strip():
        return ()

    labels: dict[str, str] = {}
    position = 0
    while position < len(raw_labels):
        match = _LABEL_PATTERN.match(raw_labels, position)
        if match is None:
            raise ValueError(f"invalid labels on line {line_number}: {raw_labels!r}")
        name = match.group("name")
        if name in labels:
            raise ValueError(f"duplicate label {name!r} on line {line_number}")
        labels[name] = _unescape_label(match.group("value"))
        position = match.end()
        while position < len(raw_labels) and raw_labels[position].isspace():
            position += 1

    return tuple(sorted(labels.items()))


def _parse_float(raw_value: str) -> float:
    normalized = raw_value.lower()
    if normalized in {"inf", "+inf"}:
        return math.inf
    if normalized == "-inf":
        return -math.inf
    if normalized == "nan":
        return math.nan
    return float(raw_value)


def parse_prometheus_text(text: str) -> tuple[MetricSample, ...]:
    """Parse samples while ignoring HELP, TYPE, UNIT, and blank lines."""
    samples: list[MetricSample] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid Prometheus sample on line {line_number}: {raw_line!r}")
        samples.append(
            MetricSample(
                name=match.group("name"),
                labels=_parse_labels(match.group("labels"), line_number=line_number),
                value=_parse_float(match.group("value")),
                timestamp=(
                    float(match.group("timestamp"))
                    if match.group("timestamp") is not None
                    else None
                ),
            )
        )
    return tuple(samples)


def index_samples(
    samples: Iterable[MetricSample],
) -> dict[tuple[str, tuple[tuple[str, str], ...]], MetricSample]:
    """Index a snapshot and reject duplicate metric/label combinations."""
    indexed: dict[tuple[str, tuple[tuple[str, str], ...]], MetricSample] = {}
    for sample in samples:
        if sample.key in indexed:
            raise ValueError(f"duplicate metric sample: {sample.key!r}")
        indexed[sample.key] = sample
    return indexed


def select_samples(
    samples: Iterable[MetricSample],
    name: str,
    *,
    labels: Mapping[str, str] | None = None,
) -> tuple[MetricSample, ...]:
    """Return samples matching a name and optional label subset."""
    required_labels = {} if labels is None else dict(labels)
    return tuple(
        sample
        for sample in samples
        if sample.name == name
        and all(sample.label_dict().get(key) == value for key, value in required_labels.items())
    )


def counter_deltas(
    before: Iterable[MetricSample],
    after: Iterable[MetricSample],
    *,
    names: set[str] | None = None,
    allow_new_series: bool = False,
) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Subtract counters while making new-series handling explicit."""
    before_index = index_samples(before)
    after_index = index_samples(after)
    deltas: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    for key, after_sample in after_index.items():
        if names is not None and after_sample.name not in names:
            continue
        if key not in before_index:
            if not allow_new_series:
                raise ValueError(f"counter has no baseline sample: {key!r}")
            before_value = 0.0
        else:
            before_value = before_index[key].value
        if after_sample.value < before_value:
            raise ValueError(f"counter reset detected for {key!r}")
        deltas[key] = after_sample.value - before_value

    return deltas
