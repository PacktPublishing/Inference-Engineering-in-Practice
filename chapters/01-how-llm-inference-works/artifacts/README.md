# Chapter 1 artifacts

Small artifacts are committed so readers can inspect the exact inputs behind
the manuscript's plots without rerunning a GPU job.

## Deterministic local artifacts

- `kv-cache-growth.csv`
- `kv-cache-growth.png`

Regenerate them from the repository root:

```bash
uv run python chapters/01-how-llm-inference-works/code/kv_cache.py \
  --output-dir chapters/01-how-llm-inference-works/artifacts
```

## GPU artifacts

The approved one-shot L4 experiment produced:

- `environment.json`
- `prompt.json`
- `generation.json`
- `generation-timings.csv`
- `cache-latency.png`

The environment manifest records the exact hardware, libraries, dtype, model
ID, immutable model revision, and random seed. GPU artifacts should not be
silently regenerated because measured values are hardware- and version-specific.

The timing sweep generates 16 tokens from synthetic repeated contexts of 64,
256, and 1,024 tokens. It is a mechanism demonstration, not a model-serving
benchmark: one request, one process, no batching, and one recorded run per
configuration after warm-up. At 1,024 initial tokens, caching reduced summed forward-pass
time from 1.164 seconds to 0.415 seconds (2.80x) on this run.

The manual cached loop exactly matched `model.generate()`. BF16 cached and
uncached paths need not be bitwise identical because their kernel shapes and
floating-point reduction orders differ; the real-prompt outputs shared the
first 17 newly generated tokens before greedy choices diverged.

These intervals include prefill and the subsequent model forward calls. They
exclude token selection, concatenation, tokenization, network time, and queueing.
The chart generator now labels this boundary explicitly; the original recorded
PNG and numerical data are retained as historical evidence.
