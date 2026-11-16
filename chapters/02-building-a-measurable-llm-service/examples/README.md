# Small measured examples

These files support the book's calculations without a full execution archive.

- [measurements.csv](measurements.csv): all 288 measured terminal rows, projected
  onto the fields used for arithmetic. Each engine has 144 measured successes.
- [comparison.csv](comparison.csv): 12 summaries, each from 24 requests.
- [engine-comparison.png](engine-comparison.png): those summaries as a plot.
- [stream.jsonl](stream.jsonl): all nine recorded events for vLLM smoke request
  `r0002`, including submitted payload, headers, parsed SSE, DONE and terminal.
  It is a parsed-event record, not a raw network capture.

## Provenance and interpretation

Runs: `vllm-20261116T065829Z` and `sglang-20261116T070238Z`, executed at source
commit `2f4a4972bf7ab866fdf7f56ae09bc3f530911b37`. Each used one NVIDIA L40S, four CPU cores, 16 GiB RAM, BF16
Qwen/Qwen2.5-1.5B-Instruct at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Full controls and launch instructions
are in the [chapter README](../README.md) and [configuration](../experiment.json).

vLLM 0.29.0 used Transformers 5.16.1; SGLang 0.5.19 used Transformers 5.12.1.
Both reported PyTorch 2.13.0+cu130 and HTTPX 0.28.1. Client and API server shared
one container over loopback HTTP. No serving framework was deployed. Context
was 1,024, engine sequence limit 64, memory fraction 0.75, prefix reuse disabled,
with normal compilation enabled. All 24 prompts warmed up before each condition.
Three repeats alternated concurrency 1 and 4. Full runs had 291 records each;
this CSV excludes warmups, smoke checks and the deliberate cancellation probe.

`started_s` and `ended_s` are seconds from the run origin. `first_content_s` and
`last_content_s` are elapsed seconds from the individual request submission.
Token counts come from server usage. All rows have complete usage; all 72
uppercase checks per engine pass. Explanation checks require only nonempty text,
not semantic correctness. `output_check_passed` uses `True`, `False`, or blank
(unknown). Stream times are elapsed seconds from that request's submission.

The full original runs passed event reconstruction, schedule, preparation and
comparison checks before projection. Every projected summary was checked against
the original summary. Full logs and executed-source snapshots are kept outside
the public tree. These compact files cannot independently revalidate omitted
warmups, deployment controls, all event streams, or remote cleanup.

## Recalculate

From the repository root, with a fresh output directory:

```bash
uv run python chapters/02-building-a-measurable-llm-service/code/reproduce.py \
  chapters/02-building-a-measurable-llm-service/examples/measurements.csv \
  --output artifacts/tmp/chapter02-comparison
```

This uses the book example's five-second objective and L40S plot label. For new
runs use `compare.py`, which checks full records and deployment metadata. Do not
use the compact-example command to certify another hardware/workload comparison.

A condition's window spans its earliest submission to its latest terminal
outcome. Successful request rate divides protocol-complete requests by that
window. Counted token rate divides their server-reported output tokens by the
same window. Output correctness and usage coverage remain separate checks.

Increasing concurrency raised throughput and median first-content latency in
both setups. These are small descriptive samples, including the packaged API
frontends and different dependencies. They do not isolate an engine mechanism
or establish production capacity. Actual generated output lengths differ.
