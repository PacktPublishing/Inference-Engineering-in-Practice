# Chapter 2 references

Accessed 2026-08-23 unless stated otherwise.

## Primary papers and official documentation

- Kwon et al., “Efficient Memory Management for Large Language Model Serving
  with PagedAttention”
  - https://arxiv.org/abs/2309.06180
  - Use for: historical allocation problem, PagedAttention design, and reported
    benchmark context
  - Verification: do not generalize historical utilization or speedup figures
    to current engines and workloads
- vLLM V1 guide
  - https://docs.vllm.ai/en/latest/usage/v1_guide/
  - Use for: V1 architecture, unified scheduling, and feature status
- vLLM optimization and tuning
  - https://docs.vllm.ai/en/latest/configuration/optimization/
  - Use for: chunked prefill and `max_num_batched_tokens` trade-offs
- vLLM PagedAttention design
  - https://docs.vllm.ai/en/stable/design/paged_attention/
  - Use for: block layout and attention-kernel implementation details
- vLLM automatic prefix caching
  - https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
  - Use for: exact-prefix reuse behavior and workload limitations
- vLLM prefix-caching design
  - https://docs.vllm.ai/en/latest/design/prefix_caching/
  - Use for: block hashing, allocation, reference counts, and eviction
- vLLM production metrics
  - https://docs.vllm.ai/en/latest/usage/metrics/
  - Use for: Prometheus metric names and semantics
- vLLM per-request metrics
  - https://docs.vllm.ai/en/latest/features/per_request_metrics/
  - Use for: request-level timings and documented overhead/limitations
- vLLM benchmark CLI
  - https://docs.vllm.ai/en/latest/benchmarking/cli/
  - Use for: official client-side benchmark definitions and cross-checks
- vLLM releases
  - https://github.com/vllm-project/vllm/releases
  - Use for: exact release notes and dependency compatibility
- Modal high-performance LLM inference
  - https://modal.com/docs/guide/high-performance-llm-inference
  - Use for: current vLLM server patterns, scaling, and cold-start guidance
- Modal web servers
  - https://modal.com/docs/guide/webhooks
  - Use for: exposing a subprocess server and binding to `0.0.0.0`

## Secondary teaching resources

- DeepLearning.AI and Red Hat, “Fast & Efficient LLM Inference with vLLM”
  - https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm
  - Use for: pedagogical sequence and workload ideas
  - Verification: independently check claims and APIs against current primary
    sources; do not copy course wording or managed-environment assumptions

## Version policy

vLLM moves quickly. Code and artifacts must record the exact vLLM, PyTorch,
Triton, CUDA, driver, model, tokenizer, and server configuration used. Prose
should describe stable mechanisms and label version-specific flags explicitly.
