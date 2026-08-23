import json
from pathlib import Path

from vllm_local_artifacts import write_local_artifacts

CHAPTER_DIR = Path(__file__).parents[1]
ARTIFACT_DIR = CHAPTER_DIR / "artifacts"


def test_committed_tabular_artifacts_match_the_generator(tmp_path) -> None:
    write_local_artifacts(tmp_path)

    for name in (
        "scheduler-simulation.json",
        "scheduler-timeline.csv",
        "allocator-simulation.json",
        "allocator-records.csv",
    ):
        assert (ARTIFACT_DIR / name).read_bytes() == (tmp_path / name).read_bytes()


def test_committed_simulation_claims_are_reconstructable() -> None:
    scheduler = json.loads(
        (ARTIFACT_DIR / "scheduler-simulation.json").read_text(encoding="utf-8")
    )
    for policy in ("static", "continuous"):
        result = scheduler[policy]
        active_slots = sum(
            request_id is not None
            for row in result["timeline"]
            for request_id in row
        )
        assert active_slots == result["active_slot_steps"]
        assert result["tokens_processed"] == sum(
            request["decode_steps"] for request in scheduler["requests"]
        )
        assert result["utilization"] == (
            result["active_slot_steps"] / result["capacity_slot_steps"]
        )

    allocator = json.loads(
        (ARTIFACT_DIR / "allocator-simulation.json").read_text(encoding="utf-8")
    )
    for policy in ("contiguous", "paged"):
        result = allocator[policy]
        assert (
            result["used_slots"] + result["wasted_slots"] + result["free_slots"]
            == result["total_slots"]
        )


def test_committed_charts_are_nonempty_png_files() -> None:
    for name in ("scheduler-simulation.png", "allocator-simulation.png"):
        image = (ARTIFACT_DIR / name).read_bytes()
        assert image.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(image) > 10_000
