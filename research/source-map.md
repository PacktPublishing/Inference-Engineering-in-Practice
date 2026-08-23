# Book-wide source map

This is a research index, not manuscript prose. Each chapter should prefer
primary papers and official documentation for technical claims. Secondary
resources can suggest experiments and explanations, but their claims must be
independently checked and their wording and code must not be copied.

## Core references

### *Inference Engineering* by Philip Kiely

- Local reference: `Inference Engineering.pdf`
- Published by Baseten Books, 2026
- Useful for: the runtime/infrastructure/tooling framing, model mechanics,
  hardware, engines, optimization techniques, and production concerns
- Verification rule: qualify broad statements such as “prefill is
  compute-bound” and “decode is memory-bound” with workload conditions

### 100 Days of LLM Inference

- URL: https://github.com/elizabetht/100-days-of-inference
- Useful for: experiment ideas covering tokenization, attention, KV caching,
  arithmetic intensity, engines, and production systems
- Chapter 1 candidates: Days 1–6
- Verification rule: recalculate complexity, KV-cache memory with
  `num_key_value_heads`, and roofline examples before reuse

## Engine and course resources

### Fast & Efficient LLM Inference with vLLM

- URL:
  https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm
- Provider: DeepLearning.AI and Red Hat
- Intended use:
  - Chapter 2: workload ideas for continuous batching, PagedAttention, prefix
    caching, metrics, and OpenAI-compatible serving
  - Chapter 8: workload normalization and engine comparison
- Verification rule: independently verify all APIs, defaults, metrics, and
  performance claims against the pinned vLLM release and primary sources
- Primary-source follow-ups:
  - PagedAttention paper: https://arxiv.org/abs/2309.06180
  - vLLM V1 guide: https://docs.vllm.ai/en/latest/usage/v1_guide/
  - vLLM tuning:
    https://docs.vllm.ai/en/latest/configuration/optimization/
  - vLLM metrics: https://docs.vllm.ai/en/latest/usage/metrics/
  - vLLM releases: https://github.com/vllm-project/vllm/releases

### Efficient Inference with SGLang: Text and Image Generation

- URL:
  https://www.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation
- Provider: DeepLearning.AI, LMSYS/SGLang, and RadixArk
- Intended use:
  - Chapter 3: SGLang serving, RadixAttention, HiCache, speculative decoding,
    and text-generation structured outputs
  - Chapter 8: normalized comparison and distributed-serving synthesis
- Exclusion: image-generation and diffusion material are outside this book
- Primary-source follow-ups: current SGLang paper, documentation,
  implementation, and release notes

### TensorRT-LLM

- Documentation: https://nvidia.github.io/TensorRT-LLM/
- Intended use:
  - Chapter 4: NVIDIA-optimized engine construction and serving
  - Chapter 6: native quantization paths and hardware-specific kernels
  - Chapter 8: normalized comparison and scale-out synthesis
- Verification rule: record TensorRT-LLM, CUDA, driver, GPU architecture, model
  conversion, engine-build, and runtime versions together

## Dedicated-topic primary sources

### Structured outputs and constrained decoding

- XGrammar paper: https://arxiv.org/abs/2411.15100
- vLLM structured outputs:
  https://docs.vllm.ai/en/latest/features/structured_outputs/
- SGLang structured outputs:
  https://docs.sglang.ai/advanced_features/structured_outputs.html
- TensorRT-LLM guided decoding:
  https://nvidia.github.io/TensorRT-LLM/latest/features/guided-decoding.html
- Intended use: engine previews in Chapters 2–4 and framework-neutral treatment
  in Chapter 5
- Verification rule: measure structural validity and semantic correctness
  separately

### Quantization

- Intended use: engine-native previews in Chapters 2–4 and dedicated serving
  treatment in Chapter 6
- Primary-source set to maintain per experiment:
  - Numerical-format specification or hardware documentation
  - Quantization-method paper
  - Artifact/conversion documentation
  - Engine/backend/kernel support documentation
- Verification rule: a format name alone does not establish artifact,
  engine, kernel, or hardware compatibility, quality, or speedup

### Speculative decoding

- Intended use: practical examples in Chapters 2–4 and deep synthesis in
  Chapter 8
- Verification rule: report draft method/model, acceptance rate, batch/load,
  memory cost, and latency/throughput regime; preserve regressions as evidence

## Per-source research record

For each source used in a chapter, record:

- Access date
- Relevant lesson, section, or page
- Claim or experiment it informed
- Primary source used to verify it
- Intended chapter and section
- Attribution or citation required
