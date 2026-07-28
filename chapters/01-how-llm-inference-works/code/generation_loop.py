"""Manual autoregressive generation with and without a KV cache."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

TokenSelector = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class GenerationResult:
    """Generated IDs plus timings and the final cache, when present."""

    token_ids: torch.Tensor
    step_seconds: list[float]
    past_key_values: Any | None = None

    @property
    def prefill_seconds(self) -> float:
        return self.step_seconds[0]

    @property
    def decode_seconds(self) -> list[float]:
        return self.step_seconds[1:]


def greedy_select(logits: torch.Tensor) -> torch.Tensor:
    """Select one token per batch row and retain a sequence dimension."""
    return logits.argmax(dim=-1, keepdim=True)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_forward(
    model: Any,
    *,
    device: torch.device,
    **model_inputs: Any,
) -> tuple[Any, float]:
    _synchronize(device)
    started = time.perf_counter()
    outputs = model(**model_inputs)
    _synchronize(device)
    return outputs, time.perf_counter() - started


def _should_stop(next_token: torch.Tensor, eos_token_id: int | None) -> bool:
    return eos_token_id is not None and bool(torch.all(next_token == eos_token_id))


def _validated_model_kwargs(model_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    extras = dict(model_kwargs or {})
    reserved = {"input_ids", "attention_mask", "past_key_values", "use_cache"}
    if overlap := reserved.intersection(extras):
        names = ", ".join(sorted(overlap))
        raise ValueError(f"model_kwargs cannot override managed inputs: {names}")
    return extras


@torch.inference_mode()
def generate_without_cache(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    select_token: TokenSelector = greedy_select,
    eos_token_id: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> GenerationResult:
    """Reprocess the complete sequence on every generation step."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, tokens]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    generated = input_ids.clone()
    step_seconds: list[float] = []
    extra_model_inputs = _validated_model_kwargs(model_kwargs)

    # snippet:uncached-loop
    for _ in range(max_new_tokens):
        attention_mask = torch.ones_like(generated)
        outputs, elapsed = _timed_forward(
            model,
            device=generated.device,
            input_ids=generated,
            attention_mask=attention_mask,
            use_cache=False,
            **extra_model_inputs,
        )
        next_token = select_token(outputs.logits[:, -1, :])
        generated = torch.cat((generated, next_token), dim=-1)
        step_seconds.append(elapsed)
        if _should_stop(next_token, eos_token_id):
            break
    # end-snippet

    return GenerationResult(token_ids=generated, step_seconds=step_seconds)


@torch.inference_mode()
def generate_with_cache(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    select_token: TokenSelector = greedy_select,
    eos_token_id: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> GenerationResult:
    """Prefill once, then pass only the newest token while reusing past K/V."""
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, tokens]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    generated = input_ids.clone()
    model_input = input_ids
    attention_mask = torch.ones_like(input_ids)
    past_key_values: Any | None = None
    step_seconds: list[float] = []
    extra_model_inputs = _validated_model_kwargs(model_kwargs)

    # snippet:cached-loop
    for _ in range(max_new_tokens):
        model_inputs = {
            **extra_model_inputs,
            "input_ids": model_input,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }
        outputs, elapsed = _timed_forward(
            model,
            device=generated.device,
            **model_inputs,
        )
        past_key_values = outputs.past_key_values
        next_token = select_token(outputs.logits[:, -1, :])

        generated = torch.cat((generated, next_token), dim=-1)
        attention_mask = torch.cat(
            (attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))),
            dim=-1,
        )
        model_input = next_token
        step_seconds.append(elapsed)
        if _should_stop(next_token, eos_token_id):
            break
    # end-snippet

    return GenerationResult(
        token_ids=generated,
        step_seconds=step_seconds,
        past_key_values=past_key_values,
    )


def timing_rows(
    result: GenerationResult,
    *,
    mode: str,
    prompt_tokens: int,
) -> list[dict[str, int | float | str]]:
    """Convert step timings to artifact-friendly rows."""
    return [
        {
            "mode": mode,
            "prompt_tokens": prompt_tokens,
            "step": step,
            "context_tokens": prompt_tokens + step,
            "seconds": seconds,
        }
        for step, seconds in enumerate(result.step_seconds)
    ]
