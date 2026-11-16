# Representative vLLM service run

Updated 13 September 2026. Choose settings for representative teaching and
transferability. Record resource use and observed charges when available.

## Purpose and execution boundary

Use the same client, lifecycle, workload and analysis on an ordinary Linux GPU
host, in the pinned engine container, or through the Modal adapter. The engine
owns scheduling and generation; the packaged API server owns HTTP validation
and streaming; the provisioner starts compute and returns files. No serving
framework is deployed in this lab. The client uses loopback HTTP, so Internet
latency is excluded.

The workflow requires exclusive use of port 8000. Before launch, it rejects an
occupied loopback port rather than accepting another server's health response.
The Modal container provides isolation. On a shared host, reserve the port and
do not start concurrent launchers; the availability check is not a port lock.

The first completed run used forced eager execution and two CPU cores to reduce
startup expense. Its [preserved evidence](artifacts/vllm-20260914T041032Z/README.md)
is useful as a preliminary run, but is not the representative cross-engine
baseline. The current run permits normal engine compilation and CUDA graphs,
uses four CPU cores, and allows realistic readiness time. Do not blend the two
configurations into one sample or attribute their differences to graphs alone.

## Fixed inputs and controls

- vLLM 0.29.0, official linux/amd64 image:
  `vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`.
- Qwen/Qwen2.5-1.5B-Instruct, model/tokenizer revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, BF16, one NVIDIA L4.
- Context 1024, scheduler sequence limit 64, GPU-memory fraction 0.75, seed 17.
  These are declared workload/resource controls, not a tuned capacity claim.
  Client concurrency is independently varied between 1 and 4.
- Prefix reuse explicitly disabled and verified from engine metrics, so warmup
  does not silently introduce a prefix-cache comparison.
- `--generation-config vllm`, temperature 0, one choice, maximum 64 output
  tokens; the deliberate cancellation probe permits 512 output tokens.
- 24 fixed prompts, three paired repeats, 144 measured requests, 144 warmups,
  two smoke checks and one cancellation: 291 requests with an explicit 300 cap.
  This is a finite experiment contract, not a dollar cap. Prompt length is
  checked on token IDs, with at most 256 chat-formatted tokens.

The 1.5B model keeps this chapter focused on the service contract and continues
Chapter 1's example. Larger-model behavior remains in the chapters where it
explains memory, quality or distributed execution.

## Commands

On a Linux GPU host with the pinned engine environment and prepared model:

```bash
python chapters/02-building-a-measurable-llm-service/code/service.py \
  --config chapters/02-building-a-measurable-llm-service/experiment.json \
  --model-path /path/to/prepared-model \
  --output artifacts/tmp/chapter02-local-service
```

Equivalent Modal adapter, from the repository root:

```bash
uv run modal run chapters/02-building-a-measurable-llm-service/code/modal_app.py \
  --output artifacts/tmp/chapter02-vllm-representative
```

CPU model preparation: two cores, 8 GiB, 600-second function deadline,
540-second download/check subprocess deadline. GPU: one L4, four cores,
16 GiB, 2100-second function deadline. Both functions permit 600 seconds
for container startup. Service readiness: 900 seconds; experiment: 900 seconds
with 15 seconds reserved for flushing; request: 30 seconds, idle read: 10 seconds.
The service orchestrator has a 1920-second overall deadline and stops the
server on exit. Shutdown waits 10 seconds before escalation. One active
container per function, no warm minimum or deployed endpoint.

Modal function and startup timeouts exclude scheduling time. The local adapter
therefore permits 1800 seconds for preparation and 3600 seconds for measurement,
including submission, queue time and result transfer. On failure or interruption,
it requests call cancellation and container termination, allowing 30 seconds for
that request. It saves call IDs and failures in `*-call.json` beside `adapter.json`.
These are provisioning records, not engine measurements. Cancellation acknowledgment
does not prove cleanup. Inspect the app and its tasks; stop the app if necessary.
If submission fails before returning a call ID, remote execution is unknown.
Use the app ID printed by the Modal CLI to stop and inspect that app.
Forced termination can prevent the remote evidence volume from being committed.
See [Modal timeouts](https://modal.com/docs/guide/timeouts) and
[call cancellation](https://modal.com/docs/reference/modal.FunctionCall#cancel).

These bounds teach finite work, failure handling and cleanup. They can be
changed when an experiment needs a different workload or startup profile;
record the change. There are no hidden client retries inside a measurement.
Diagnosed reruns preserve their own logs and configuration.

## Deployment checks and lessons retained

The first image build succeeded but Modal Function registration failed because
the image provided `python3` without the `python` command Modal expects. The
adapter now adds an alias to the existing interpreter, preserving installed
engine packages. The corrected remote run verified Python 3.12.3 and completed.

The first completed run also exposed a preparation metadata bug caused by
Transformers 5's dictionary return default. That bug was fixed by explicitly
requesting token IDs. Independent retokenization matched all 144 measured
server counts (37–44 tokens); the raw manifest and correction are both retained.

The public model downloaded without an HF token. Modal authentication is
already configured on the author's laptop. Other readers supply their own
compute access; neither account identity nor credentials belong in the client.

Sources:

- [vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.29.0)
- [Pinned engine arguments](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/engine/arg_utils.py)
- [Pinned Dockerfile](https://github.com/vllm-project/vllm/blob/v0.29.0/docker/Dockerfile)
- [Modal image requirements](https://modal.com/docs/guide/existing-images)
- [Modal CUDA support](https://modal.com/docs/guide/cuda)

## Matching SGLang deployment

SGLang 0.5.19 uses official linux/amd64 image
`lmsysorg/sglang@sha256:37bbbd3444732a464bbc68dee4fb0164e0ce9e18e2f027f3fc967f1152d3c262`.
The selected release registers Qwen2ForCausalLM and its Dockerfile uses CUDA
13.0.3. Keep the same model snapshot, BF16, one L4, four CPU cores, 16 GiB,
workload, deadlines and client topology as the representative vLLM run.

SGLang-specific controls remain visible in `engines.py`: context length 1024,
maximum running requests 64, static-memory fraction 0.75, disabled radix cache,
OpenAI sampling defaults with explicit request temperature 0, seed 17 and
metrics enabled. `/server_info` reports the resolved cache configuration.
Normal execution is retained; CUDA graph behavior follows the selected engine.

```bash
CHAPTER_ENGINE=sglang uv run modal run \
  chapters/02-building-a-measurable-llm-service/code/modal_app.py \
  --output artifacts/tmp/chapter02-sglang-representative
```

On a host/container with SGLang installed, the standalone `service.py` command
uses `--engine sglang`. The HTTP client and analysis do not change. Memory
fractions and sequence limits are controls with engine-specific semantics;
matching numeric values is not proof of identical internal allocations.
Record their effective values and compare complete serving setups.

Sources: [SGLang release](https://github.com/sgl-project/sglang/releases/tag/v0.5.19),
[pinned server arguments](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/server_args.py),
[Qwen2 implementation](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/models/qwen2.py),
[Dockerfile](https://github.com/sgl-project/sglang/blob/v0.5.19/docker/Dockerfile).

The final matched runs use a complete 24-prompt warmup pass before every
condition. The initial four-request warmup covered only the shortest task type;
the first SGLang concurrent repeat varied materially. Preserve those earlier
observations as diagnostics and use the revised, matched procedure for the
main comparison. Shutdown now signals the parent before escalating, allowing
the engine to manage its own child processes.


The portable service now polls `/health` for readiness. For SGLang it sets
`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`: the endpoint returns 503 while
server status is Starting, and does not generate health requests after startup.
A captured failed attempt showed `/v1/models` succeeding before internal warmup
finished, followed by a smoke idle timeout. The corrected deployment uses the
recommended `sglang serve` entrypoint. Measurement deadlines are unchanged.
