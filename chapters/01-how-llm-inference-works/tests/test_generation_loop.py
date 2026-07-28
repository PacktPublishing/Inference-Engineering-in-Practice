from functools import partial

import pytest
import torch
from generation_loop import generate_with_cache, generate_without_cache
from kv_cache import cache_summary
from sampling import sample_next_token
from transformers import AutoModelForCausalLM, Qwen2Config


@pytest.fixture
def tiny_qwen() -> torch.nn.Module:
    torch.manual_seed(13)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=63,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def test_cached_and_uncached_greedy_generation_match(tiny_qwen: torch.nn.Module) -> None:
    prompt = torch.tensor([[1, 7, 11, 5]])

    uncached = generate_without_cache(tiny_qwen, prompt, max_new_tokens=6)
    cached = generate_with_cache(tiny_qwen, prompt, max_new_tokens=6)

    assert torch.equal(uncached.token_ids, cached.token_ids)
    assert len(uncached.step_seconds) == 6
    assert len(cached.step_seconds) == 6
    assert cache_summary(cached.past_key_values)["num_layers"] == 2


def test_generation_does_not_modify_the_prompt(tiny_qwen: torch.nn.Module) -> None:
    prompt = torch.tensor([[1, 3, 5]])
    original = prompt.clone()
    generate_with_cache(tiny_qwen, prompt, max_new_tokens=2)
    assert torch.equal(prompt, original)


def test_generation_forwards_safe_model_options(tiny_qwen: torch.nn.Module) -> None:
    prompt = torch.tensor([[1, 3, 5]])
    options = {"logits_to_keep": 1}

    uncached = generate_without_cache(
        tiny_qwen,
        prompt,
        max_new_tokens=3,
        model_kwargs=options,
    )
    cached = generate_with_cache(
        tiny_qwen,
        prompt,
        max_new_tokens=3,
        model_kwargs=options,
    )

    assert torch.equal(uncached.token_ids, cached.token_ids)


def test_batched_sampler_plugs_into_generation_loop(tiny_qwen: torch.nn.Module) -> None:
    prompt = torch.tensor([[1, 7, 11, 5], [1, 3, 9, 2]])
    sampler = partial(
        sample_next_token,
        temperature=0.8,
        top_p=0.9,
        generator=torch.Generator().manual_seed(17),
    )

    result = generate_with_cache(
        tiny_qwen,
        prompt,
        max_new_tokens=4,
        select_token=sampler,
    )

    assert result.token_ids.shape == (2, 8)


@pytest.mark.parametrize(
    "function",
    [generate_with_cache, generate_without_cache],
)
def test_generation_validates_inputs(function, tiny_qwen: torch.nn.Module) -> None:
    with pytest.raises(ValueError, match="shape"):
        function(tiny_qwen, torch.tensor([1, 2]), max_new_tokens=1)
    with pytest.raises(ValueError, match="positive"):
        function(tiny_qwen, torch.tensor([[1, 2]]), max_new_tokens=0)
    with pytest.raises(ValueError, match="managed inputs"):
        function(
            tiny_qwen,
            torch.tensor([[1, 2]]),
            max_new_tokens=1,
            model_kwargs={"use_cache": False},
        )
