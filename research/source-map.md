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

## Course resources for later chapters

### Fast & Efficient LLM Inference with vLLM

- URL:
  https://www.deeplearning.ai/courses/fast-and-efficient-llm-inference-with-vllm
- Provider: DeepLearning.AI and Red Hat
- Candidate chapters:
  - Chapter 2: benchmarking and model-quality evaluation
  - Chapter 4: vLLM, continuous batching, PagedAttention, and prefix caching
  - Chapter 5: realistic traffic generation and engine comparison
  - Chapter 8: weight and activation quantization
- Primary-source follow-ups: vLLM paper/docs, PagedAttention paper, LLM
  Compressor docs, GuideLLM docs, and lm-evaluation-harness docs

### Efficient Inference with SGLang: Text and Image Generation

- URL:
  https://www.deeplearning.ai/courses/efficient-inference-with-sglang-text-and-image-generation
- Provider: DeepLearning.AI, LMSYS/SGLang, and RadixArk
- Candidate chapters:
  - Chapter 5: SGLang serving and engine comparison
  - Chapter 9: RadixAttention and cross-request prefix reuse
  - Chapter 11: multi-GPU parallelism
  - Chapter 12: inference for diffusion and image generation
- Primary-source follow-ups: SGLang paper/docs, RadixAttention paper and
  implementation, and the relevant diffusion-model papers

## Per-source research record

For each source used in a chapter, record:

- Access date
- Relevant lesson, section, or page
- Claim or experiment it informed
- Primary source used to verify it
- Intended chapter and section
- Attribution or citation required
