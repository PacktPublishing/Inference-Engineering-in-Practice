"""Controlled next-token selection from a model's logits."""

from __future__ import annotations

import torch


def _as_batched_logits(logits: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if logits.ndim == 1:
        return logits.unsqueeze(0), True
    if logits.ndim == 2:
        return logits, False
    raise ValueError("logits must have shape [vocabulary] or [batch, vocabulary]")


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Select one highest-logit token for an unbatched or batched input."""
    batched_logits, was_unbatched = _as_batched_logits(logits)
    next_token = torch.argmax(batched_logits, dim=-1, keepdim=True)
    return next_token[0, 0] if was_unbatched else next_token


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row after temperature, top-k, and top-p filtering."""
    batched_logits, was_unbatched = _as_batched_logits(logits)
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if temperature == 0:
        return greedy_next_token(logits)
    if top_k is not None and not 1 <= top_k <= batched_logits.shape[-1]:
        raise ValueError("top_k must be between 1 and the vocabulary size")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    # snippet:sampling-core
    filtered = batched_logits.float() / temperature

    if top_k is not None:
        threshold = torch.topk(filtered, top_k, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None and top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    probabilities = torch.softmax(filtered, dim=-1)
    next_token = torch.multinomial(probabilities, 1, generator=generator)
    # end-snippet
    return next_token[0, 0] if was_unbatched else next_token


def demo() -> None:
    """Show how the same logits produce deterministic and sampled choices."""
    vocabulary = ["fast", "cheap", "reliable", "purple", "banana"]
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.5])

    print("Greedy:", vocabulary[greedy_next_token(logits).item()])
    for temperature in (0.5, 1.0, 2.0):
        generator = torch.Generator().manual_seed(11)
        samples = [
            vocabulary[
                sample_next_token(
                    logits,
                    temperature=temperature,
                    top_k=5,
                    top_p=0.95,
                    generator=generator,
                ).item()
            ]
            for _ in range(8)
        ]
        print(f"temperature={temperature:.1f}:", ", ".join(samples))


if __name__ == "__main__":
    demo()
