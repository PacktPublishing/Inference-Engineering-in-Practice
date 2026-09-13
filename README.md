# LLM Inference Engineering in Practice

Runnable companion code for the Packt book *LLM Inference Engineering in
Practice*.

The manuscript is written code first: each mechanism is implemented, tested,
and measured before the explanation is finalized. Chapters contain the
essential snippets and representative output; this repository contains the
complete programs, tests, and reproducibility metadata.

## Chapters

- [Chapter 1: Following an LLM Request Through the Inference Stack](chapters/01-how-llm-inference-works)

Chapter 2, *Building a Measurable LLM Service*, will be written and implemented
from scratch under the revised concept-first plan.

## Local setup

The project uses Python 3.12 and an exact `uv` lockfile. On macOS with a
cloud-managed Documents folder, first run `sh scripts/setup_macos.sh` to keep
dependencies local. See [the environment guidance](DEVELOPMENT.md#keeping-python-dependencies-local-on-macos).

For ordinary local checkouts:

```bash
uv sync
source .venv/bin/activate
pytest
```

Most correctness examples run locally without a GPU. GPU-only experiments are
clearly labeled and use a thin Modal runner; the inference code itself is not
coupled to Modal.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the author workflow. Readers should
follow the requirements and commands in each chapter folder.
