# Chapter 2 figure inventory

Conceptual figures will be created in Excalidraw and visually verified. Measured
charts will be generated from artifact data by chapter code.

## Planned Excalidraw figures

1. Chapter 1 generation loop versus a multi-request serving engine
2. vLLM V1 API server, scheduler, KV manager, workers, kernels, and GPU
3. Static batching versus continuous batching timeline
4. Contiguous KV reservation versus paged block allocation
5. Logical block table mapped to physical KV blocks
6. Prefix-block hashing, reuse, reference counts, and eviction
7. Chunked prefill mixed with active decode requests
8. Client, server, and GPU measurement layers
9. Modal server lifecycle and artifact lineage
10. Workload/SLA-to-configuration decision flow

## Planned generated charts

- `../artifacts/scheduler-simulation.png`: static versus continuous batching
  timeline generated from the local simulator
- `../artifacts/allocator-simulation.png`: equal-capacity contiguous versus
  paged KV allocation generated from the local simulator
- Planned after the approved GPU run:
  - Concurrency versus request and token throughput
  - P50/P95/P99 TTFT, TPOT, and end-to-end latency
  - TTFT–ITL tuning frontier for chunked prefill
  - Shared/unique and cold/warm prefix-cache comparisons
  - KV-cache utilization and request-admission curve
  - GPU-seconds and estimated cost per million tokens

Scene URLs, artifact dependencies, and final captions will be added after each
figure is created and reviewed.
