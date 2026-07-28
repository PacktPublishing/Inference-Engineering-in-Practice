import pytest
import torch
from attention import (
    causal_attention_mask,
    expand_key_value_heads,
    scaled_dot_product_attention,
)
from torch.nn import functional as F


def test_reference_attention_matches_pytorch_sdpa() -> None:
    torch.manual_seed(3)
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 2, 5, 8)
    value = torch.randn(2, 2, 5, 8)

    actual, _ = scaled_dot_product_attention(query, key, value)
    expanded_key, expanded_value = expand_key_value_heads(key, value, query_heads=4)
    expected = F.scaled_dot_product_attention(
        query,
        expanded_key,
        expanded_value,
        is_causal=True,
    )

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_causal_attention_never_looks_ahead() -> None:
    query = torch.ones(1, 1, 4, 2)
    key = torch.ones(1, 1, 4, 2)
    value = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)

    _, weights = scaled_dot_product_attention(query, key, value)

    assert torch.count_nonzero(torch.triu(weights[0, 0], diagonal=1)) == 0
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))


def test_decode_mask_allows_the_complete_past() -> None:
    mask = causal_attention_mask(query_length=1, key_value_length=7)
    assert mask.shape == (1, 7)
    assert torch.all(mask)


def test_gqa_matches_explicitly_repeated_key_value_heads() -> None:
    torch.manual_seed(5)
    query = torch.randn(1, 6, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)

    actual, _ = scaled_dot_product_attention(query, key, value)
    repeated_key, repeated_value = expand_key_value_heads(key, value, query_heads=6)
    expected, _ = scaled_dot_product_attention(query, repeated_key, repeated_value)

    torch.testing.assert_close(actual, expected)


def test_invalid_gqa_head_ratio_is_rejected() -> None:
    key = torch.zeros(1, 2, 2, 4)
    value = torch.zeros_like(key)
    with pytest.raises(ValueError, match="divisible"):
        expand_key_value_heads(key, value, query_heads=3)
