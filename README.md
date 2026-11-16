# Inference Engineering in Practice

Runnable examples for the book by Dewang Sultania. Read the chapters for the
explanations and essential outputs; use this repository to inspect, change, and
run the programs.

- [Chapter 1: Following an LLM Request Through the Inference Stack](chapters/01-how-llm-inference-works/README.md)
- [Chapter 2: Building a Measurable LLM Service](chapters/02-building-a-measurable-llm-service/README.md)

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run pytest
```

The tests need no GPU, credentials, or model download. Each chapter separates
local exercises from optional GPU execution. Engine packages belong in their
own GPU environments, not the local test environment.

Small teaching datasets and figures are included. Write new runs under ignored
`artifacts/tmp/`; raw logs, repeated source copies, and execution archives do
not belong in Git. See [development checks](DEVELOPMENT.md) before contributing.
