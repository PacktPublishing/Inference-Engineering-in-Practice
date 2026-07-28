"""A small, inspectable implementation of causal scaled dot-product attention."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def causal_attention_mask(
    query_length: int,
    key_value_length: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a mask for queries occupying the newest positions in the KV sequence."""
    if query_length <= 0 or key_value_length <= 0:
        raise ValueError("sequence lengths must be positive")
    if query_length > key_value_length:
        raise ValueError("query_length cannot exceed key_value_length")

    first_query_position = key_value_length - query_length
    query_positions = torch.arange(
        first_query_position,
        key_value_length,
        device=device,
    )
    key_positions = torch.arange(key_value_length, device=device)
    return key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)


def expand_key_value_heads(
    key: torch.Tensor,
    value: torch.Tensor,
    query_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat shared K/V heads so a reference implementation can evaluate GQA."""
    key_value_heads = key.shape[-3]
    if value.shape[-3] != key_value_heads:
        raise ValueError("key and value must have the same number of heads")
    if query_heads % key_value_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")

    repeats = query_heads // key_value_heads
    return (
        key.repeat_interleave(repeats, dim=-3),
        value.repeat_interleave(repeats, dim=-3),
    )


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention and return both its output and inspectable weights."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [batch, heads, tokens, dim]")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("query, key, and value must have the same batch size")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key head dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")

    key, value = expand_key_value_heads(key, value, query.shape[-3])

    # snippet:attention-core
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = query @ key.transpose(-2, -1) * scale

    if causal:
        mask = causal_attention_mask(
            query_length=query.shape[-2],
            key_value_length=key.shape[-2],
            device=query.device,
        )
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    output = weights @ value
    # end-snippet
    return output, weights


def demo() -> None:
    """Compare the reference implementation with PyTorch's optimized SDPA."""
    torch.manual_seed(7)
    query = torch.randn(1, 4, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)

    output, weights = scaled_dot_product_attention(query, key, value)
    expanded_key, expanded_value = expand_key_value_heads(key, value, query.shape[-3])
    expected = F.scaled_dot_product_attention(
        query,
        expanded_key,
        expanded_value,
        is_causal=True,
    )

    print("Q shape:", tuple(query.shape))
    print("K/V shape before GQA expansion:", tuple(key.shape))
    print("One causal attention matrix:")
    print(weights[0, 0].round(decimals=3))
    print("Maximum difference from torch SDPA:", (output - expected).abs().max().item())


if __name__ == "__main__":
    demo()
