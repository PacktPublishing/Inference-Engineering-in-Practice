import torch
from experiments import run_generation_experiment
from transformers import AutoModelForCausalLM, Qwen2Config


class TinyTokenizer:
    eos_token_id = 63
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages and not tokenize and add_generation_prompt
        return "tiny prompt"

    def __call__(self, text, *, return_tensors):
        assert text == "tiny prompt" and return_tensors == "pt"
        input_ids = torch.tensor([[1, 7, 11, 5]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids.tolist())


def test_real_model_orchestration_with_tiny_random_model() -> None:
    torch.manual_seed(21)
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
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()

    result = run_generation_experiment(
        TinyTokenizer(),
        model,
        answer_max_new_tokens=4,
        benchmark_context_lengths=(4, 8),
        benchmark_new_tokens=2,
    )

    assert result["answer"]["manual_cached_matches_generate"]
    assert result["answer"]["tokens_equal"]
    assert len(result["benchmark"]["comparisons"]) == 2
    assert len(result["timings"]) == 8
