# Chapter 1 figure sequence

Each diagram answers a reader question before the corresponding code or
measurement. The new diagrams have local editable SVG sources and matching
high-resolution PNG exports embedded in the tracked Word manuscript. Labels
use text as well as color; diagrams are conceptual, not measured traces.

| Figure | Question and teaching purpose | Source or evidence |
| --- | --- | --- |
| 1.1 | Which responsibility handles the API, and which handles model work? A built-in engine server can provide both. | [SVG](01-serving-and-engine.svg) · [PNG](01-serving-and-engine.png) |
| 1.2 | How do a formatted conversation and token IDs become one next-token prediction? | Existing publisher figure; [recorded prompt](../artifacts/prompt.json) |
| 1.3 | Why can a new query read earlier K/V without recomputing them? | [SVG](03-attention-history.svg) · [PNG](03-attention-history.png) |
| 1.4 | Which model call predicts the first token, and what enters later calls? | [SVG](04-prefill-and-decode.svg) · [PNG](04-prefill-and-decode.png) |
| 1.5 | How do request length and batch size change ideal tensor storage? | [CSV](../artifacts/kv-cache-growth.csv) · [plot](../artifacts/kv-cache-growth.png) |
| 1.6 | Why do 41 prompt IDs plus 32 selected IDs leave only 72 cached positions? | [SVG](06-cache-state.svg) · [PNG](06-cache-state.png) · [record](../artifacts/generation.json) |
| 1.7 | What did this one recorded cached/uncached comparison actually time? | [raw intervals](../artifacts/generation-timings.csv) · [plot](../artifacts/cache-latency.png) |
| 1.8 | Which evidence should we inspect first when outputs differ? | [SVG](08-diagnostic-path.svg) · [PNG](08-diagnostic-path.png) |

Figures 1.5 and 1.7 retain the original plotted evidence. Figure 1.7's original
latency label means summed synchronized model forward intervals, including
prefill; it excludes token selection, concatenation, network time, and queueing.
The generator now labels this boundary explicitly for future exports.

## Visual conventions

Use a plain white background, readable short labels, and directional arrows.
Blue denotes input or serving work; orange denotes new decode work or engine
work; green denotes retained state or evidence. Every region also carries a
text label so color is never the only cue. A figure should explain one mechanism
at manuscript size. Captions state the operational consequence, and prose
introduces what readers should notice before presenting a listing.

The local SVG files are the editable source for the replacement diagrams.
The following earlier Excalidraw references are preserved as design history;
figures 1.1, 1.3, 1.4, and 1.6 now use the local assets above.

## Earlier figure inventory

# Chapter 1 figure inventory

Conceptual figures are maintained as editable Excalidraw scenes. Empirical
charts are generated from committed experiment artifacts.

## Excalidraw figures

Collection: `chapter_1` (`9npGAe1eFyX`)

1. **Three layers shape every inference decision** — application requirements
   flow through runtime, infrastructure, and tooling concerns before becoming
   a production inference service.
   [Open scene](https://app.excalidraw.com/s/8RQ78d1ZePv/8hXrXwaXovw)
2. **From prompt text to one next token** — chat template, token IDs,
   embeddings, transformer blocks, logits, and token selection.
   [Open scene](https://app.excalidraw.com/s/8RQ78d1ZePv/5Fze2rSakCA)
3. **Causal scaled dot-product attention** — Q/K/V projection, scaled scores,
   causal masking, softmax, and the weighted value sum.
   [Open scene](https://app.excalidraw.com/s/8RQ78d1ZePv/9eT8QW6yHus)
4. **One request has two very different phases** — parallel prompt prefill
   followed by a sequential one-token decode loop.
   [Open scene](https://app.excalidraw.com/s/8RQ78d1ZePv/3hvUwLlmIl)
5. **Why the KV cache grows with every token** — runtime growth and the
   architecture-aware sizing formula, including GQA's KV-head count.
   [Open scene](https://app.excalidraw.com/s/8RQ78d1ZePv/49jLHFsXkzd)

All five scenes were visually checked after creation. Export final SVG or PNG
assets from Excalidraw when preparing the Word manuscript so labels remain
sharp at print resolution.

## Generated charts

- `artifacts/cache-latency.png`: cached versus uncached per-step latency
- `artifacts/kv-cache-growth.png`: cache memory versus context and batch
