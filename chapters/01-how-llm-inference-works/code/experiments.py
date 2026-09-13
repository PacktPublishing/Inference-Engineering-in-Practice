"""Platform-neutral orchestration for the Chapter 1 real-model experiment."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from generation_loop import (
    GenerationResult,
    generate_with_cache,
    generate_without_cache,
    timing_rows,
)
from kv_cache import cache_summary
from prompt_to_logits import DEFAULT_PROMPT, inspect_prompt, prepare_inputs

MODEL_FORWARD_OPTIONS = {"logits_to_keep": 1}


def _new_token_prefix_length(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    prompt_tokens: int,
) -> int:
    left_tokens = left[0, prompt_tokens:].tolist()
    right_tokens = right[0, prompt_tokens:].tolist()
    return next(
        (
            index
            for index, (left_id, right_id) in enumerate(
                zip(left_tokens, right_tokens, strict=False)
            )
            if left_id != right_id
        ),
        min(len(left_tokens), len(right_tokens)),
    )


def _repeat_to_length(input_ids: torch.Tensor, length: int) -> torch.Tensor:
    repeats = (length + input_ids.shape[-1] - 1) // input_ids.shape[-1]
    return input_ids.repeat(1, repeats)[:, :length]


def _comparison(
    uncached: GenerationResult,
    cached: GenerationResult,
    *,
    prompt_tokens: int,
) -> dict[str, Any]:
    uncached_seconds = sum(uncached.step_seconds)
    cached_seconds = sum(cached.step_seconds)
    return {
        "tokens_equal": torch.equal(uncached.token_ids, cached.token_ids),
        "common_new_token_prefix": _new_token_prefix_length(
            uncached.token_ids,
            cached.token_ids,
            prompt_tokens=prompt_tokens,
        ),
        "uncached_total_seconds": uncached_seconds,
        "cached_total_seconds": cached_seconds,
        "speedup": uncached_seconds / cached_seconds,
    }


@torch.inference_mode()
def _generate_answer(
    tokenizer: Any,
    model: Any,
    input_ids: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any], GenerationResult]:
    eos_token_id = tokenizer.eos_token_id
    uncached = generate_without_cache(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        model_kwargs=MODEL_FORWARD_OPTIONS,
    )
    cached = generate_with_cache(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        model_kwargs=MODEL_FORWARD_OPTIONS,
    )
    library_output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    new_tokens = cached.token_ids[:, input_ids.shape[-1] :]

    return (
        {
            "max_new_tokens": max_new_tokens,
            **_comparison(
                uncached,
                cached,
                prompt_tokens=input_ids.shape[-1],
            ),
            "manual_cached_matches_generate": torch.equal(
                cached.token_ids,
                library_output,
            ),
            "generated_token_ids": new_tokens[0].tolist(),
            "generated_text": tokenizer.decode(new_tokens[0], skip_special_tokens=True),
            "final_cache": cache_summary(cached.past_key_values),
        },
        cached,
    )


@torch.inference_mode()
def _benchmark_contexts(
    model: Any,
    seed_ids: torch.Tensor,
    *,
    context_lengths: tuple[int, ...],
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    comparisons = []
    all_rows = []

    for context_length in context_lengths:
        context = _repeat_to_length(seed_ids, context_length)

        # Warm both paths at this tensor shape before recording a single run.
        generate_without_cache(
            model,
            context,
            max_new_tokens=2,
            model_kwargs=MODEL_FORWARD_OPTIONS,
        )
        generate_with_cache(
            model,
            context,
            max_new_tokens=2,
            model_kwargs=MODEL_FORWARD_OPTIONS,
        )

        uncached = generate_without_cache(
            model,
            context,
            max_new_tokens=max_new_tokens,
            model_kwargs=MODEL_FORWARD_OPTIONS,
        )
        cached = generate_with_cache(
            model,
            context,
            max_new_tokens=max_new_tokens,
            model_kwargs=MODEL_FORWARD_OPTIONS,
        )
        comparisons.append(
            {
                "context_tokens": context_length,
                **_comparison(
                    uncached,
                    cached,
                    prompt_tokens=context_length,
                ),
            }
        )
        for mode, result in (("uncached", uncached), ("cached", cached)):
            all_rows.extend(
                {
                    **row,
                    "benchmark_context_tokens": context_length,
                    "max_new_tokens": max_new_tokens,
                }
                for row in timing_rows(
                    result,
                    mode=mode,
                    prompt_tokens=context_length,
                )
            )

    return comparisons, all_rows


@torch.inference_mode()
def run_generation_experiment(
    tokenizer: Any,
    model: Any,
    *,
    prompt: str = DEFAULT_PROMPT,
    answer_max_new_tokens: int = 64,
    benchmark_context_lengths: tuple[int, ...] = (64, 256, 1024),
    benchmark_new_tokens: int = 16,
) -> dict[str, Any]:
    """Generate a real answer, then benchmark short and long synthetic contexts."""
    device = next(model.parameters()).device
    _, inputs = prepare_inputs(tokenizer, prompt, device)
    input_ids = inputs["input_ids"]

    model(**inputs, use_cache=True, **MODEL_FORWARD_OPTIONS)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    answer, _ = _generate_answer(
        tokenizer,
        model,
        input_ids,
        inputs,
        max_new_tokens=answer_max_new_tokens,
    )
    comparisons, all_rows = _benchmark_contexts(
        model,
        input_ids,
        context_lengths=benchmark_context_lengths,
        max_new_tokens=benchmark_new_tokens,
    )

    return {
        "prompt_tokens": input_ids.shape[-1],
        "answer": answer,
        "benchmark": {
            "uses_synthetic_repeated_context": True,
            "context_lengths": list(benchmark_context_lengths),
            "new_tokens_per_run": benchmark_new_tokens,
            "comparisons": comparisons,
        },
        "timings": all_rows,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }


def write_experiment_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    """Persist small results returned by the remote GPU function."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "llm-inference-engineering-matplotlib"),
    )
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("environment", "prompt", "generation"):
        payload = result[key]
        if key == "generation":
            payload = {name: value for name, value in payload.items() if name != "timings"}
        (output_dir / f"{key}.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    rows = result["generation"]["timings"]
    csv_path = output_dir / "generation-timings.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    context_lengths = result["generation"]["benchmark"]["context_lengths"]
    for mode in ("uncached", "cached"):
        totals = [
            sum(
                row["seconds"]
                for row in rows
                if row["mode"] == mode and row["benchmark_context_tokens"] == length
            )
            * 1000
            for length in context_lengths
        ]
        axis.plot(context_lengths, totals, marker="o", label=mode)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Initial context tokens")
    new_tokens = result["generation"]["benchmark"]["new_tokens_per_run"]
    axis.set_ylabel(f"Summed forward time for {new_tokens} new tokens (ms)")
    gpu = result["environment"].get("gpu", "recorded device")
    axis.set_title(f"Manual greedy generation on {gpu}")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cache-latency.png", dpi=180)
    plt.close(figure)


def run_full_experiment(tokenizer: Any, model: Any, environment: dict[str, Any]) -> dict:
    """Collect all real-model observations in one model load."""
    return {
        "environment": environment,
        "prompt": inspect_prompt(tokenizer, model, DEFAULT_PROMPT),
        "generation": run_generation_experiment(tokenizer, model),
    }
