import csv
import json

from vllm_local_artifacts import write_local_artifacts


def test_local_artifact_set_is_complete_and_internally_consistent(tmp_path) -> None:
    paths = write_local_artifacts(tmp_path)

    assert {path.name for path in paths} == {
        "scheduler-simulation.json",
        "scheduler-timeline.csv",
        "scheduler-simulation.png",
        "allocator-simulation.json",
        "allocator-records.csv",
        "allocator-simulation.png",
    }
    assert all(path.exists() for path in paths)

    scheduler = json.loads(
        (tmp_path / "scheduler-simulation.json").read_text(encoding="utf-8")
    )
    assert scheduler["static"]["tokens_processed"] == 24
    assert scheduler["continuous"]["tokens_processed"] == 24
    assert (
        scheduler["continuous"]["total_steps"]
        < scheduler["static"]["total_steps"]
    )
    assert (
        scheduler["continuous"]["utilization"]
        > scheduler["static"]["utilization"]
    )

    allocator = json.loads(
        (tmp_path / "allocator-simulation.json").read_text(encoding="utf-8")
    )
    assert allocator["contiguous"]["admitted_requests"] == 2
    assert allocator["paged"]["admitted_requests"] == 4
    assert allocator["paged"]["wasted_slots"] == 24

    for image_name in ("scheduler-simulation.png", "allocator-simulation.png"):
        assert (tmp_path / image_name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_local_tabular_artifacts_are_deterministic(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_local_artifacts(first)
    write_local_artifacts(second)

    for name in (
        "scheduler-simulation.json",
        "scheduler-timeline.csv",
        "allocator-simulation.json",
        "allocator-records.csv",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    with (first / "scheduler-timeline.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    scheduler = json.loads(
        (first / "scheduler-simulation.json").read_text(encoding="utf-8")
    )
    expected_rows = 3 * (
        scheduler["static"]["total_steps"]
        + scheduler["continuous"]["total_steps"]
    )
    assert len(rows) == expected_rows
