"""CPU-only weight preparation and chat-template checks inside the engine image."""

import argparse
import hashlib
import json
from pathlib import Path


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
    lengths = {
        p["id"]: len(
            tokenizer.apply_chat_template(p["messages"], tokenize=True, add_generation_prompt=True)
        )
        for p in config["prompts"]
    }
    if max(lengths.values()) > config["prompt_token_limit"]:
        raise ValueError("A formatted prompt exceeds the configured token limit")
    metadata = {
        "model": config["model"],
        "revision": config["model_revision"],
        "prompt_tokens": lengths,
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
