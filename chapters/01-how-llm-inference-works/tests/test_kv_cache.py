import pytest
import torch
from kv_cache import (
    QWEN_2_5_1_5B,
    ModelShape,
    cache_summary,
    kv_cache_bytes,
)
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer


def test_qwen_cache_size_uses_kv_heads_not_query_heads() -> None:
    bytes_per_token = kv_cache_bytes(
        QWEN_2_5_1_5B,
        batch_size=1,
        sequence_length=1,
        bytes_per_element=2,
    )
    full_context = kv_cache_bytes(
        QWEN_2_5_1_5B,
        batch_size=1,
        sequence_length=32768,
        bytes_per_element=2,
    )

    assert bytes_per_token == 28 * 1024
    assert full_context == 896 * 1024**2


def test_gqa_cache_is_smaller_than_mha_by_head_ratio() -> None:
    mha_shape = ModelShape(
        num_layers=QWEN_2_5_1_5B.num_layers,
        num_attention_heads=12,
        num_key_value_heads=12,
        hidden_size=1536,
    )
    gqa_bytes = kv_cache_bytes(
        QWEN_2_5_1_5B,
        batch_size=4,
        sequence_length=2048,
        bytes_per_element=2,
    )
    mha_bytes = kv_cache_bytes(
        mha_shape,
        batch_size=4,
        sequence_length=2048,
        bytes_per_element=2,
    )
    assert mha_bytes == 6 * gqa_bytes


def test_cache_summary_reads_transformers_dynamic_layers() -> None:
    cache = DynamicCache()
    layer = DynamicLayer()
    layer.keys = torch.zeros(1, 2, 3, 4, dtype=torch.bfloat16)
    layer.values = torch.zeros_like(layer.keys)
    layer.is_initialized = True
    cache.layers.append(layer)

    summary = cache_summary(cache)

    assert summary["num_layers"] == 1
    assert summary["layers"][0]["key_shape"] == [1, 2, 3, 4]
    assert summary["total_bytes"] == 96


def test_invalid_cache_dimensions_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        kv_cache_bytes(
            QWEN_2_5_1_5B,
            batch_size=0,
            sequence_length=1,
            bytes_per_element=2,
        )
