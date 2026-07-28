import csv
import json
from pathlib import Path

import pytest
from kv_cache import ModelShape, kv_cache_bytes
from model_config import MODEL_ID, MODEL_REVISION, TORCH_VERSION, TRANSFORMERS_VERSION

CHAPTER_DIR = Path(__file__).parents[1]
ARTIFACT_DIR = CHAPTER_DIR / "artifacts"


def _read_json(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def test_environment_manifest_matches_pinned_configuration() -> None:
    environment = _read_json("environment.json")
    assert environment["model_id"] == MODEL_ID
    assert environment["model_revision"] == MODEL_REVISION
    assert environment["torch"].startswith(TORCH_VERSION)
    assert environment["transformers"] == TRANSFORMERS_VERSION
    assert environment["gpu"] == "NVIDIA L4"


def test_inspected_prompt_cache_matches_architecture_formula() -> None:
    prompt = _read_json("prompt.json")
    model = prompt["model_shape"]
    shape = ModelShape(
        num_layers=model["layers"],
        num_attention_heads=model["attention_heads"],
        num_key_value_heads=model["key_value_heads"],
        hidden_size=model["hidden_size"],
        head_dim=model["head_dim"],
    )
    expected = kv_cache_bytes(
        shape,
        batch_size=1,
        sequence_length=len(prompt["token_ids"]),
        bytes_per_element=2,
    )
    assert prompt["cache"]["total_bytes"] == expected


def test_timing_csv_reconstructs_reported_totals() -> None:
    generation = _read_json("generation.json")
    with (ARTIFACT_DIR / "generation-timings.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))

    for comparison in generation["benchmark"]["comparisons"]:
        context = comparison["context_tokens"]
        for mode in ("cached", "uncached"):
            total = sum(
                float(row["seconds"])
                for row in rows
                if row["mode"] == mode and int(row["benchmark_context_tokens"]) == context
            )
            assert total == pytest.approx(comparison[f"{mode}_total_seconds"])

    assert generation["answer"]["manual_cached_matches_generate"]
