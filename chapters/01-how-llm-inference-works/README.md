# Chapter 1: How LLM Inference Works

This folder contains the complete companion code for Chapter 1. The chapter
manuscript includes the essential snippets and explanations; these files add
the full programs, tests, generated artifacts, and reproducibility metadata.

Complete repository:
https://github.com/PacktPublishing/Inference-Engineering-in-Practice

## What runs where

- `attention.py`, `sampling.py`, and KV-cache sizing tests: CPU, no account
  required
- Real-model prompt/logit and generation examples: CPU/MPS where practical
- CUDA timing and cache inspection: one on-demand NVIDIA L4 through Modal

Modal is only the GPU sandbox. None of the inference logic depends on Modal.
A Modal account is required only for the labeled GPU experiment.

## Setup

From the repository root:

```bash
uv sync
source .venv/bin/activate
```

Run local checks:

```bash
uv run ruff check .
uv run pytest
```

Run the local, deterministic examples:

```bash
uv run python chapters/01-how-llm-inference-works/code/attention.py
uv run python chapters/01-how-llm-inference-works/code/sampling.py
uv run python chapters/01-how-llm-inference-works/code/kv_cache.py
```

## GPU experiment

Authenticate the Modal CLI once:

```bash
uv run modal setup
```

Run the consolidated GPU experiment:

```bash
uv run modal run \
  chapters/01-how-llm-inference-works/code/modal_app.py
```

This command incurs usage on one L4. Review current Modal pricing and account
credits first. It creates no deployed endpoint and leaves no warm GPU running.

## Layout

- `chapter.md`: standalone manuscript draft
- `code/`: complete runnable examples
- `tests/`: fast correctness and snippet-consistency tests
- `artifacts/`: small, curated outputs from verified runs
- `figures/`: figure inventory and Excalidraw scene references
- `references.md`: sources for factual and API claims
