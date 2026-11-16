# Chapter 1: Following an LLM Request Through the Inference Stack

Follow one request from its chat template to token IDs, next-token scores,
prefill, decode, and retained KV state. The figures introduce each mechanism;
the code makes the request and its state inspectable. This is a single-request
walkthrough, not a production serving benchmark.

## Start here

1. Read the [figure sequence](figures/README.md).
2. Calculate ideal cache storage locally, without a GPU or model download:

   ```bash
   python3 chapters/01-how-llm-inference-works/code/kv_cache.py
   ```

   The Qwen example reports **28 KiB per cached token** and **896 MiB** for
   one 32,768-token request. These are ideal K/V tensor bytes, not total GPU memory.
3. Inspect [prompt.json](artifacts/prompt.json) alongside the prompt-to-token
   diagram. Identify the rendered conversation, 41 input tokens, and cache shapes.
4. Read `generate_with_cache` in [generation_loop.py](code/generation_loop.py).
   Track the full prompt on the first call, one new input on later calls, and
   the growing attention mask. Compare with `generate_without_cache` only to
   see the repeated work.
5. Reconstruct the recorded timing totals using [the artifact notes](artifacts/README.md).
   Use the chapter's failure cases to distinguish formatting, generation-policy,
   cache-position, and measurement problems.

## Code map

| File | Reader question |
| --- | --- |
| [prompt_to_logits.py](code/prompt_to_logits.py) | What did the model actually receive and predict? |
| [generation_loop.py](code/generation_loop.py) | What changes after each selected token? |
| [kv_cache.py](code/kv_cache.py) | How much state does one request retain? |
| [experiments.py](code/experiments.py) | What evidence distinguishes saved computation from a timing claim? |
| [model_config.py](code/model_config.py) | Which immutable model and package versions were used? |
| [modal_app.py](code/modal_app.py) | How can the recorded experiment be repeated on a GPU? |

## Local verification

From the repository root, install the locked Python 3.12 environment:

```bash
uv sync --locked
uv run ruff check chapters/01-how-llm-inference-works
uv run pytest chapters/01-how-llm-inference-works/tests
```

The tests exercise a tiny random model created locally, masks and cache state,
matching generation settings, recorded cache arithmetic, and reconstructed timing
totals. They need no model downloads, credentials, or GPU.

## Optional model execution

On a CUDA host with the locked PyTorch/Transformers environment and a compatible
NVIDIA driver, inspect one prompt locally:

```bash
uv run python chapters/01-how-llm-inference-works/code/prompt_to_logits.py --device cuda
```

This downloads the pinned model and prints prompt/token/logit/cache observations.
It does not run the full generation/timing sweep. The model and package pins are
in [model_config.py](code/model_config.py). Complete timing collection uses
`run_full_experiment` in `experiments.py`; the inference functions have no Modal
dependency.

The optional Modal wrapper rents a GPU and runs that full experiment:

```bash
uv run modal setup
uv run modal run chapters/01-how-llm-inference-works/code/modal_app.py
```

New results default to ignored `artifacts/tmp/chapter01`. Use `--output-dir` for
another fresh directory. The wrapper refuses to overwrite an existing output
location. Keep the original small measured examples when experimenting.
