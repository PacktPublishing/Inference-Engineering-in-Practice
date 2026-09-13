"""Architecture-aware KV-cache sizing and inspection."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ModelShape:
    """The model dimensions that determine KV-cache size."""

    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int | None = None

    @property
    def resolved_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_config(cls, config: Any) -> ModelShape:
        """Build a shape from a Hugging Face model configuration."""
        return cls(
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=getattr(
                config,
                "num_key_value_heads",
                config.num_attention_heads,
            ),
            hidden_size=config.hidden_size,
            head_dim=getattr(config, "head_dim", None),
        )


QWEN_2_5_1_5B = ModelShape(
    num_layers=28,
    num_attention_heads=12,
    num_key_value_heads=2,
    hidden_size=1536,
)


def kv_cache_bytes(
    shape: ModelShape,
    *,
    batch_size: int,
    sequence_length: int,
    bytes_per_element: int,
) -> int:
    """Calculate the ideal, unpadded K/V tensor storage in bytes."""
    if batch_size <= 0 or sequence_length <= 0 or bytes_per_element <= 0:
        raise ValueError("batch, sequence length, and element size must be positive")

    # snippet:kv-cache-formula
    elements = (
        2  # one tensor for keys and one for values
        * batch_size
        * sequence_length
        * shape.num_layers
        * shape.num_key_value_heads
        * shape.resolved_head_dim
    )
    return elements * bytes_per_element
    # end-snippet


def human_bytes(number: int) -> str:
    """Format a byte count using binary units."""
    value = float(number)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _layer_key_values(layer: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    import torch

    for key_name, value_name in (("keys", "values"), ("key_cache", "value_cache")):
        key = getattr(layer, key_name, None)
        value = getattr(layer, value_name, None)
        if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
            return key, value
    return None


def cache_summary(cache: Any) -> dict[str, Any]:
    """Describe tensors in a modern Transformers Cache or a legacy tuple cache."""
    import torch

    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    if hasattr(cache, "layers"):
        for layer in cache.layers:
            pair = _layer_key_values(layer)
            if pair is not None:
                pairs.append(pair)
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        pairs.extend(zip(cache.key_cache, cache.value_cache, strict=True))
    elif isinstance(cache, (tuple, list)):
        for layer in cache:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                key, value = layer[:2]
                if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
                    pairs.append((key, value))

    layers = []
    total_bytes = 0
    for index, (key, value) in enumerate(pairs):
        layer_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
        total_bytes += layer_bytes
        layers.append(
            {
                "layer": index,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "dtype": str(key.dtype),
                "bytes": layer_bytes,
            }
        )

    return {"num_layers": len(layers), "total_bytes": total_bytes, "layers": layers}


def cache_sweep(
    shape: ModelShape = QWEN_2_5_1_5B,
    *,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    sequence_lengths: tuple[int, ...] = (128, 512, 2048, 8192, 32768),
    bytes_per_element: int = 2,
) -> list[dict[str, int]]:
    """Calculate a small cache-memory grid without allocating those tensors."""
    return [
        {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "bytes_per_element": bytes_per_element,
            "cache_bytes": kv_cache_bytes(
                shape,
                batch_size=batch_size,
                sequence_length=sequence_length,
                bytes_per_element=bytes_per_element,
            ),
        }
        for batch_size in batch_sizes
        for sequence_length in sequence_lengths
    ]


def write_artifacts(output_dir: Path) -> None:
    """Write the deterministic cache sweep and its plot."""
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "llm-inference-engineering-matplotlib"),
    )
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = cache_sweep()

    csv_path = output_dir / "kv-cache-growth.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for batch_size in sorted({row["batch_size"] for row in rows}):
        selected = [row for row in rows if row["batch_size"] == batch_size]
        axis.plot(
            [row["sequence_length"] for row in selected],
            [row["cache_bytes"] / 1024**3 for row in selected],
            marker="o",
            label=f"batch={batch_size}",
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Cached tokens per request")
    axis.set_ylabel("Ideal KV-cache storage (GiB)")
    axis.set_title("Qwen2.5-1.5B BF16 KV-cache growth")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "kv-cache-growth.png", dpi=180)
    plt.close(figure)


def demo() -> None:
    per_token = kv_cache_bytes(
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
    print("Qwen2.5-1.5B KV heads:", QWEN_2_5_1_5B.num_key_value_heads)
    print("Head dimension:", QWEN_2_5_1_5B.resolved_head_dim)
    print("BF16 cache per token:", human_bytes(per_token))
    print("BF16 cache at 32,768 tokens, batch 1:", human_bytes(full_context))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    demo()
    if args.output_dir is not None:
        write_artifacts(args.output_dir)


if __name__ == "__main__":
    main()
