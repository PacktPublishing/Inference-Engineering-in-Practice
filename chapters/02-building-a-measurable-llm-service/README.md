# Chapter 2: Building a Measurable LLM Service

Follow a request through an HTTP API, distinguish stream events from model
tokens, and measure latency and throughput without hiding failed requests.
The client and packaged API server share one host over loopback HTTP. The lab
has one engine replica and no serving framework or public endpoint.

## Start without a GPU

From the repository root:

```bash
uv sync --locked
uv run pytest chapters/02-building-a-measurable-llm-service/tests -q
uv run python chapters/02-building-a-measurable-llm-service/code/fixture_demo.py \
  --output artifacts/tmp/chapter02-fixtures
uv run python chapters/02-building-a-measurable-llm-service/code/reproduce.py \
  chapters/02-building-a-measurable-llm-service/examples/measurements.csv \
  --output artifacts/tmp/chapter02-comparison
```

Choose fresh output directories. The fixture report shows how counting events
as tokens or removing failures produces misleading rates. Its text and usage
are invented; its elapsed times are not model performance.

The [small measured dataset](examples/README.md) contains 288 measured request
rows, a selected stream, and the book's comparison plot. `reproduce.py`
recalculates the CSV summaries and plot. It does not validate omitted raw event
traces or the complete execution. Full run logs are not included in Git.

## Code map

| Program | Purpose |
| --- | --- |
| [client.py](code/client.py), [records.py](code/records.py) | Parse ordinary/SSE responses; retain timing, usage, partial text and failure outcomes. |
| [experiment.py](code/experiment.py) | Run smoke checks, full-workload warmup, bounded workers and a cancellation probe. |
| [analyze.py](code/analyze.py), [evidence.py](code/evidence.py) | Check new request/event records and calculate rates and distributions. |
| [compare.py](code/compare.py), [reproduce.py](code/reproduce.py) | Compare complete new runs, or recalculate the compact book dataset. |
| [fixture_demo.py](code/fixture_demo.py) | Demonstrate two measurement errors without a model. |
| [prepare_model.py](code/prepare_model.py), [engines.py](code/engines.py) | Prepare pinned model files and construct the two engine launch commands. |
| [service.py](code/service.py), [server_process.py](code/server_process.py) | Start, check, measure and stop a local server with bounded waits. |
| [modal_app.py](code/modal_app.py), [modal_calls.py](code/modal_calls.py) | Optional compute, transfer and bounded remote-call adapter. |

## Run on a Linux GPU host

Use one NVIDIA L40S to match the saved example, four CPU cores, 16 GiB host
memory, and a compatible NVIDIA driver/container runtime. The selected model is
Qwen/Qwen2.5-1.5B-Instruct, BF16, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

Use a separate engine environment. The tested linux/amd64 image references are:

- vLLM 0.29.0: `vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`
- SGLang 0.5.19: `lmsysorg/sglang@sha256:37bbbd3444732a464bbc68dee4fb0164e0ce9e18e2f027f3fc967f1152d3c262`

For example, start an interactive vLLM container from the repository root:

```bash
docker run --rm -it --gpus '"device=0"' --ipc=host \
  --cpus=4 --memory=16g -v "$PWD:/work" -w /work --entrypoint /bin/bash \
  vllm/vllm-openai@sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b
```

Inside that environment, pin the client dependency as in the measured runs,
then prepare the model and run the service:

```bash
python3 -m pip install --no-deps httpx==0.28.1
python3 chapters/02-building-a-measurable-llm-service/code/prepare_model.py \
  chapters/02-building-a-measurable-llm-service/experiment.json \
  artifacts/tmp/chapter02-model
python3 chapters/02-building-a-measurable-llm-service/code/service.py \
  --config chapters/02-building-a-measurable-llm-service/experiment.json \
  --model-path artifacts/tmp/chapter02-model \
  --gpu L40S:1 --cpu-cores 4 --memory-mib 16384 \
  --output artifacts/tmp/chapter02-vllm
```

For SGLang, use its image and add `--engine sglang` to the service command.
Prepare the model in that environment too; tokenizer and
runtime dependencies are part of the serving setup. Use a different result
directory. The same platform-neutral service code handles both engines.

The resource arguments declare the allocation for comparison; set them to your
actual host/container limits. They do not provision or enforce resources. The
service also saves observed `nvidia-smi` output and small source-file hashes.
Check the hardware observation against the declaration before comparing runs.

The [configuration](experiment.json) supplies 24 prompts, concurrency 1 and 4,
three alternating repeats, and 24 warmup requests before each condition. It
schedules 291 requests: 144 measured, 144 warmup, two smoke checks and one probe.
The launch controls use context 1,024, scheduler limit 64, memory fraction 0.75,
seed 17 and disabled prefix reuse. Normal engine compilation remains enabled.
Requests use temperature 0 and at most 64 output tokens; the probe permits 512.

Reserve loopback port 8000 for the service. Startup waits up to 900 seconds for
release-specific health, followed by ordinary and streaming smoke checks.
Requests have a 30-second total deadline and a 10-second idle timeout. The run
has a 900-second deadline including 15 seconds for saving records. The service
has an overall 1,920-second bound and a 10-second cooperative shutdown wait
before escalation. Failed runs retain partial records. There are no hidden
measurement retries.

## Use an existing endpoint or inspect new results

An already running compatible endpoint can use the client directly:

```bash
uv run python chapters/02-building-a-measurable-llm-service/code/experiment.py \
  --base-url http://127.0.0.1:8000/v1 \
  --config chapters/02-building-a-measurable-llm-service/experiment.json \
  --output artifacts/tmp/chapter02-endpoint
```

Record that endpoint's hardware, model revision, tokenizer and effective launch
controls separately. The model name returned by HTTP is not sufficient.

For a new service run, analyze its `run/` directory:

```bash
uv run python chapters/02-building-a-measurable-llm-service/code/analyze.py \
  artifacts/tmp/chapter02-vllm/run --output artifacts/tmp/chapter02-analysis
uv run python chapters/02-building-a-measurable-llm-service/code/compare.py \
  artifacts/tmp/chapter02-vllm/run artifacts/tmp/chapter02-sglang/run \
  --output artifacts/tmp/chapter02-new-comparison
```

For an endpoint-only run, pass its output directory directly to `analyze.py`.
Comparison requires the preparation/deployment metadata saved by `service.py`.
It checks workload, hardware declarations, templates, prompt counts and event
records. Completion means the schedule finished; inspect failure fractions,
usage coverage and output lengths alongside speed.

First-content time ends at nonempty text, not necessarily one model token.
Completion includes client response cleanup. Token throughput uses server usage
and the full condition window, including failures. The five-second objective
combines completion with exact uppercase or nonempty explanation checks; it is
not a semantic quality evaluation. A closed client socket does not establish
engine cancellation. The client is intended for these local APIs; its payload
limits do not bound every possible unfinished HTTP/SSE framing buffer.

## Optional Modal execution

Modal supplies compute and file transfer. It calls the same local service
function. These commands rent GPUs and require a configured Modal account:

```bash
CHAPTER_GPU=L40S uv run modal run chapters/02-building-a-measurable-llm-service/code/modal_app.py \
  --output artifacts/tmp/chapter02-vllm-modal
CHAPTER_GPU=L40S CHAPTER_ENGINE=sglang uv run modal run chapters/02-building-a-measurable-llm-service/code/modal_app.py \
  --output artifacts/tmp/chapter02-sglang-modal
```

The adapter defaults to L4 unless `CHAPTER_GPU` is set. Use matching hardware
for comparisons. It bounds submission and waiting (1,800 seconds for preparation,
3,600 for measurement), and attempts cancellation on failure. Inspect remote
app state after interruption: cancellation acknowledgement alone does not prove
cleanup. Image building before the entrypoint is outside those caller deadlines.
