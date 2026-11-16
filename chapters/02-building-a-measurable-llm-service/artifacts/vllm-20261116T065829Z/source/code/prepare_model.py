"""CPU-only weight preparation and chat-template checks inside the engine image."""

import argparse
import hashlib
import json
from pathlib import Path


def prompt_lengths(tokenizer, prompts: list[dict], limit: int) -> dict[str, int]:
    lengths = {}
    for prompt in prompts:
        ids = tokenizer.apply_chat_template(
            prompt["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ValueError("Expected a single flat sequence of token IDs")
        lengths[prompt["id"]] = len(ids)
    if not lengths or max(lengths.values()) > limit:
        raise ValueError(
            "A formatted prompt exceeds the configured token limit or none were supplied"
        )
    return lengths


def validate_preparation(config: dict, metadata: dict, *, require_prompt_hash: bool = True) -> None:
    """Reject a prepared directory whose recorded identity differs from the run."""
    if (
        metadata.get("model") != config["model"]
        or metadata.get("revision") != config["model_revision"]
    ):
        raise ValueError("Prepared model identity differs from the experiment")
    template = metadata.get("chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError("Prepared chat template is missing")
    if hashlib.sha256(template.encode()).hexdigest() != metadata.get("chat_template_sha256"):
        raise ValueError("Prepared chat template hash differs")
    counts = metadata.get("prompt_tokens", {})
    if set(counts) != {p["id"] for p in config["prompts"]}:
        raise ValueError("Prepared prompt IDs differ from the workload")
    if any(
        type(n) is not int or not 0 < n <= config["prompt_token_limit"] for n in counts.values()
    ):
        raise ValueError("Prepared prompt counts are invalid")
    # New preparations bind the exact messages, not only their IDs and lengths.
    from records import content_hash

    if (require_prompt_hash or "prompts_sha256" in metadata) and (
        metadata.get("prompts_sha256") != content_hash(config["prompts"])
    ):
        raise ValueError("Prepared prompt content differs from the workload")


def prepare(config_path: Path, model_path: Path):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    config = json.loads(config_path.read_text())
    snapshot_download(
        config["model"],
        revision=config["model_revision"],
        local_dir=model_path,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"],
        max_workers=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    lengths = prompt_lengths(tokenizer, config["prompts"], config["prompt_token_limit"])
    from records import content_hash

    metadata = {
        "model": config["model"],
        "revision": config["model_revision"],
        "prompt_tokens": lengths,
        "prompts_sha256": content_hash(config["prompts"]),
        "chat_template": tokenizer.chat_template,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
    }
    (model_path / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("model_directory", type=Path)
    args = parser.parse_args()
    prepare(args.config, args.model_directory)
