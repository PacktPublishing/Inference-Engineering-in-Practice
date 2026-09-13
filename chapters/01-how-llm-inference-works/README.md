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

Attention and sampling are explained visually and through the model/library
interfaces. The standalone implementations have been removed from the Chapter 1
reader path and code. Generation uses the existing plain greedy selector.

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

## Optional GPU reproduction

The chapter can be read using the committed evidence. Loading the real model
requires its weights; the optional Modal command also incurs cloud usage.
Review the exact run and costs using [DEVELOPMENT.md](../../DEVELOPMENT.md).

```bash
uv run modal setup
uv run modal run chapters/01-how-llm-inference-works/code/modal_app.py
```

Keep the original measured artifacts when experimenting. Write new results to a
separate location and record their environment. The inference modules remain
independent of the Modal wrapper.

The publisher manuscript lives in Word. The ignored `chapter.md` is a reading
copy synchronized with the revised manuscript, with its retained snippets checked
against executable code.
