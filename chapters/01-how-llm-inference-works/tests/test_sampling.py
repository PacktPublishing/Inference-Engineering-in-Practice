import pytest
import torch
from sampling import greedy_next_token, sample_next_token


def test_greedy_selects_largest_logit() -> None:
    logits = torch.tensor([-1.0, 3.0, 2.0])
    assert greedy_next_token(logits).item() == 1
    assert sample_next_token(logits, temperature=0).item() == 1


def test_batched_sampling_returns_one_token_per_row() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0], [4.0, 1.0, 2.0]])
    generator = torch.Generator().manual_seed(9)

    greedy = greedy_next_token(logits)
    sampled = sample_next_token(logits, top_k=1, generator=generator)

    assert greedy.shape == (2, 1)
    assert sampled.shape == (2, 1)
    assert greedy.tolist() == [[1], [0]]
    assert sampled.tolist() == [[1], [0]]


def test_top_k_one_is_deterministic() -> None:
    logits = torch.tensor([0.0, 2.0, 1.0])
    generator = torch.Generator().manual_seed(9)
    assert sample_next_token(logits, top_k=1, generator=generator).item() == 1


def test_top_p_keeps_the_first_token_that_crosses_threshold() -> None:
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
    for seed in range(10):
        generator = torch.Generator().manual_seed(seed)
        assert sample_next_token(logits, top_p=0.6, generator=generator).item() == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": -1.0}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
    ],
)
def test_invalid_sampling_arguments_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sample_next_token(torch.ones(3), **kwargs)


def test_invalid_logit_rank_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        sample_next_token(torch.ones(1, 2, 3))
