# vllm — representative full-workload warmup

Run `vllm-20260914T044208Z` completed with 144 measured successes, 144 warmup successes,
two contract smoke successes and one deliberate client cancellation. Each
condition warms up with the full 24-prompt workload, then measures that same
set. Three paired repeats compare client concurrency one and four.

The model is the pinned Qwen2.5-1.5B-Instruct snapshot, BF16, on one L4 with
four CPU cores and 16 GiB host RAM. Normal engine execution is retained;
prefix reuse is disabled and verified. Server-reported prompt counts match
preparation at 37–44 tokens. All successful measured outputs include usage.

See [the matched comparison](../comparison-20260914/README.md) for the table,
plot, controls and remaining differences. Regenerate this run's summaries by
passing its `run/` directory to the chapter `analyze.py`.

The original command was `modal run chapters/02-building-a-measurable-llm-service/code/modal_app.py --output artifacts/tmp/chapter02-vllm-representative`,
using the default vLLM adapter.
The exact executed source is under `source/` and matches the manifest hashes.
New reproductions use the current code and run specification. The client
experiment took 39.74 seconds; startup, preparation and engine
internal warmup are separate. This is not total billed runtime.

App `ap-dJ6sl3cU1fJO08WykBUC3G` is stopped with zero tasks. The parent-first shutdown path is
retained in the log, including any engine cleanup warnings. A client closed
stream is not by itself proof of engine resource reclamation.
