"""Trace one real prompt through tokenization, embeddings, logits, and cache."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from kv_cache import ModelShape, cache_summary
from model_config import MODEL_ID, MODEL_REVISION

DEFAULT_PROMPT = "Explain in one sentence why LLM decode is sequential."


def choose_device(requested: str = "auto") -> torch.device:
    """Choose a local accelerator when available, with an explicit override."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    *,
    device: torch.device,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
) -> tuple[Any, Any]:
    """Load the pinned tokenizer and causal language model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def prepare_inputs(tokenizer: Any, prompt: str, device: torch.device) -> tuple[str, dict]:
    """Apply the model's own chat template before tokenization."""
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    return rendered, {name: tensor.to(device) for name, tensor in encoded.items()}


@torch.inference_mode()
def inspect_prompt(tokenizer: Any, model: Any, prompt: str) -> dict[str, Any]:
    """Return JSON-serializable observations from one model forward pass."""
    device = next(model.parameters()).device

    # snippet:prompt-to-logits
    rendered, inputs = prepare_inputs(tokenizer, prompt, device)
    embeddings = model.get_input_embeddings()(inputs["input_ids"])
    outputs = model(**inputs, use_cache=True)

    next_token_logits = outputs.logits[:, -1, :]
    probabilities = torch.softmax(next_token_logits.float(), dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=5, dim=-1)
    # end-snippet

    token_ids = inputs["input_ids"][0].tolist()
    candidates = [
        {
            "token_id": token_id,
            "text": tokenizer.decode([token_id]),
            "probability": probability,
        }
        for token_id, probability in zip(
            top_ids[0].tolist(),
            top_probabilities[0].tolist(),
            strict=True,
        )
    ]
    shape = ModelShape.from_config(model.config)

    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt": prompt,
        "rendered_prompt": rendered,
        "token_ids": token_ids,
        "token_pieces": tokenizer.convert_ids_to_tokens(token_ids),
        "tensor_shapes": {
            "input_ids": list(inputs["input_ids"].shape),
            "embeddings": list(embeddings.shape),
            "logits": list(outputs.logits.shape),
        },
        "model_shape": {
            "layers": shape.num_layers,
            "hidden_size": shape.hidden_size,
            "attention_heads": shape.num_attention_heads,
            "key_value_heads": shape.num_key_value_heads,
            "head_dim": shape.resolved_head_dim,
        },
        "top_next_tokens": candidates,
        "cache": cache_summary(outputs.past_key_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    args = parser.parse_args()

    device = choose_device(args.device)
    tokenizer, model = load_model(device=device)
    print(json.dumps(inspect_prompt(tokenizer, model, args.prompt), indent=2))


if __name__ == "__main__":
    main()
