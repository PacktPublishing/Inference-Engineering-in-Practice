# Chapter 2: vLLM — Build, Observe, and Tune a Serving Engine

This folder contains the runnable companion material for Chapter 2. It starts
from Chapter 1's single-request generation loop and builds a measured,
OpenAI-compatible vLLM serving workflow.

Complete repository:
https://github.com/PacktPublishing/Inference-Engineering-in-Practice

## What runs where

- Stream parsing, metric parsing, benchmark accounting, workload construction,
  and scheduler/allocation simulations: local CPU
- Client tests: local CPU with mocked HTTP responses
- vLLM server, scheduling experiments, and GPU measurements: one on-demand
  NVIDIA GPU through Modal

The core measurement and simulation code does not depend on Modal. The Modal
module is a thin runner around a pinned `vllm serve` command.

## Local setup

From the repository root:

```bash
uv sync
source .venv/bin/activate
```

Run all local checks:

```bash
uv run ruff check .
uv run pytest
```

No local or CI test requires a GPU, model download, Modal credentials, or a
running vLLM server.

## Remote experiments

A Modal account is required for the labeled GPU experiments. Before an author
run, follow the billing, approval, and timeout safeguards in
[DEVELOPMENT.md](../../DEVELOPMENT.md).

Reader-facing commands will be added only after the pinned server path has been
executed and verified. Defining or importing the Modal app must not start
remote compute.

## Chapter questions

The code and experiments are designed to answer:

- Why does continuous batching improve throughput, and where does queueing hurt
  tail latency?
- How does paged KV-cache allocation change usable memory?
- When does automatic prefix caching reduce prefill work?
- How does chunked prefill trade time to first token against inter-token
  latency?
- Which server configuration satisfies a stated latency, throughput, memory,
  and cost target?

## Layout

- `code/`: complete clients, parsers, simulations, benchmarks, and Modal runner
- `tests/`: CPU-only correctness and artifact-contract tests
- `artifacts/`: curated raw measurements, derived data, and generated charts
- `figures/`: Excalidraw inventory and figure provenance
- `references.md`: primary papers, official documentation, and claim notes
- `chapter.md`: local ignored manuscript draft

## Current code

- `vllm_config.py`: pinned model, vLLM, image, server, and cost-boundary settings
- `vllm_modal.py`: pure `vllm serve` command builder plus an on-demand Modal web
  server definition
- `vllm_client.py` and `vllm_streaming.py`: model discovery, sync requests, SSE
  assembly, TTFT, and E2E timing
- `vllm_metrics.py`: labeled Prometheus samples, selection, and counter deltas
- `vllm_workloads.py`: deterministic request and workload manifests
- `vllm_benchmark.py`: request records, percentiles, throughput, TPOT, and cost
  normalization
- `vllm_scheduler_sim.py` and `vllm_allocator_sim.py`: inspectable educational
  models for batching and KV allocation
- `vllm_local_artifacts.py`: deterministic JSON, CSV, and PNG generation
