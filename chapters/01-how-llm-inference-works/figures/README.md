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
