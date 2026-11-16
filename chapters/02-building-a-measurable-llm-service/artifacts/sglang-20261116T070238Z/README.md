# Current-source SGLang service evidence

Run `sglang-20261116T070238Z` completed on 16 November 2026 UTC using commit
`2f4a4972bf7ab866fdf7f56ae09bc3f530911b37`. The saved `source/` files are the
bytes copied inside the executing container, including the effective configuration.

## What ran

- Qwen/Qwen2.5-1.5B-Instruct at revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16.
- SGLang 0.5.19 in the image digest recorded in `launch.json`.
- One NVIDIA L40S, four CPU cores and 16 GiB host RAM. Actual device/driver and package versions are in `launch.json`.
- Client, packaged API server and engine share one temporary container; HTTP uses loopback. No serving framework is deployed.
- Normal engine execution, context 1024, sequence limit 64, memory fraction 0.75, prefix reuse disabled.
- 24 prompts, concurrency 1 and 4, three alternating repeats, a full warmup before each condition.

All 291 terminal records are present: two smoke checks, 144 warmups, 144 measured
successes and one separate client cancellation probe. All 72 measured uppercase
checks passed. The explanation prompts have no general semantic-quality check.
Both saved metrics observations returned HTTP 200 with valid required gauges.
Prefix-cache configuration was checked separately. The service stopped its engine;
a subsequent provider query reported the app stopped, zero tasks and no live
containers. These external cleanup observations are recorded in `cleanup.json`.

## Reproduce and verify

From the repository root, with configured Modal access:

```bash
CHAPTER_GPU=L40S CHAPTER_ENGINE=sglang uv run modal run chapters/02-building-a-measurable-llm-service/code/modal_app.py \
  --output artifacts/tmp/reproduce-sglang-l40s
```

Inspect this preserved bundle offline, without a GPU or model download:

```bash
uv run python chapters/02-building-a-measurable-llm-service/code/verify_bundle.py \
  chapters/02-building-a-measurable-llm-service/artifacts/sglang-20261116T070238Z \
  --output artifacts/tmp/verify-sglang-current
```

`run/` contains the raw request/event/manifest records. The top-level summaries
and plot were derived from those records. `SHA256SUMS.json` covers the curated
files; hashes detect disagreement, not independently authenticated provenance.
Local provider call IDs and console logs remain private diagnostics.

## Limits

This is a small, short workload and a complete serving-setup observation,
including its API frontend. It is not a stable-tail, production-capacity or
isolated-kernel benchmark. Server usage supplies token counts. A stream event
is not a token. Closing the client stream does not prove per-request engine
work or KV state was reclaimed. Before/after queue gauges are aggregate snapshots.
The L40S pair is separate from the historical September L4 comparison. Two
preceding L4 attempts were stopped while queued and produced no GPU measurements.
Their wait is not evidence about engine speed. Actual billed spend was unavailable.
