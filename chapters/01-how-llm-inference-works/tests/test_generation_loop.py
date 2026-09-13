import pytest
import torch
from generation_loop import generate_with_cache, generate_without_cache
from kv_cache import cache_summary
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


def test_custom_selector_plugs_into_batched_generation(tiny_qwen: torch.nn.Module) -> None:
    prompt = torch.tensor([[1, 7, 11, 5], [1, 3, 9, 2]])
    calls = []

    def select_fixed_token(logits: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(logits.shape))
        return torch.full((logits.shape[0], 1), 7, device=logits.device)

    result = generate_with_cache(
        tiny_qwen,
        prompt,
        max_new_tokens=4,
        select_token=select_fixed_token,
    )

    assert result.token_ids.shape == (2, 8)
    assert calls == [(2, 64)] * 4
    assert torch.equal(result.token_ids[:, :4], prompt)
    assert torch.all(result.token_ids[:, 4:] == 7)


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
