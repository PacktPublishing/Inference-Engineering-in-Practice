# Chapter 1 figure sequence

Each diagram answers a reader question before the corresponding code or
measurement. The diagrams use editable SVG files. Labels
use text as well as color; diagrams are conceptual, not measured traces.

| Figure | Question and teaching purpose | Source or evidence |
| --- | --- | --- |
| 1.1 | Which responsibility handles the API, and which handles model work? A built-in engine server can provide both. | [SVG](01-serving-and-engine.svg) |
| 1.2 | How do a formatted conversation and token IDs become one next-token prediction? | [Diagram](02-prompt-to-token.png); [recorded prompt](../artifacts/prompt.json) |
| 1.3 | Why can a new query read earlier K/V without recomputing them? | [SVG](03-attention-history.svg) |
| 1.4 | Which model call predicts the first token, and what enters later calls? | [SVG](04-prefill-and-decode.svg) |
| 1.5 | How do request length and batch size change ideal tensor storage? | [CSV](../artifacts/kv-cache-growth.csv) · [plot](../artifacts/kv-cache-growth.png) |
| 1.6 | Why do 41 prompt IDs plus 32 selected IDs leave only 72 cached positions? | [SVG](06-cache-state.svg) · [record](../artifacts/generation.json) |
| 1.7 | What did this one recorded cached/uncached comparison actually time? | [raw intervals](../artifacts/generation-timings.csv) · [plot](../artifacts/cache-latency.png) |
| 1.8 | Which evidence should we inspect first when outputs differ? | [SVG](08-diagnostic-path.svg) |

Figures 1.5 and 1.7 retain the original plotted evidence. Figure 1.7's original
latency label means summed synchronized model forward intervals, including
prefill; it excludes token selection, concatenation, network time, and queueing.
The generator now labels this boundary explicitly for future exports.
