"""Keep the compact reader dataset numerically consistent with its saved summary."""

import csv
from pathlib import Path

import pytest
from reproduce import read_measurements, reproduce

EXAMPLES = Path(__file__).parents[1] / "examples"


def test_compact_example_reproduces_every_summary(tmp_path):
    actual = reproduce(EXAMPLES / "measurements.csv", tmp_path / "result")
    with (EXAMPLES / "comparison.csv").open() as file:
        expected = list(csv.DictReader(file))
    assert len(actual) == len(expected) == 12
    for row, saved in zip(actual, expected, strict=True):
        assert row.keys() == saved.keys()
        for key, value in row.items():
            if isinstance(value, (int, float)):
                assert value == pytest.approx(float(saved[key]), rel=1e-12, abs=1e-12)
            else:
                assert (str(value) if value is not None else "") == saved[key]
    assert (tmp_path / "result/engine-comparison.png").is_file()


@pytest.mark.parametrize(
    "field,value", [("ended_s", "nan"), ("started_s", ""), ("run_id", "another-run")]
)
def test_invalid_projection_is_not_silently_plotted(tmp_path, field, value):
    with (EXAMPLES / "measurements.csv").open() as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = reader.fieldnames
    rows[0][field] = value
    path = tmp_path / "bad.csv"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        read_measurements(path)
