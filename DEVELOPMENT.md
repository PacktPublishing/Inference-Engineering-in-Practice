# Author development workflow

This file describes the book-development workflow. It is not manuscript
content and its spending limit is not guidance for readers.

## Git ownership

The author owns repository history. Agents may edit and validate the working
tree. Commit or push only when the author explicitly requests it for the current
work. Follow [AGENTS.md](AGENTS.md), the canonical guidance for coding, writing,
evidence and remote execution.

## Bootstrap

Use Python 3.12 and the locked dependencies. On macOS, run
`sh scripts/setup_macos.sh` first; see the environment guidance below.

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```

Run `uv run modal setup` only when preparing an approved GPU experiment.
Authentication itself does not run GPU code.

## Current implementation scope

Chapters 1 and 2 now have implemented code. The former Chapter 2 implementation,
shared serving benchmarks, simulations, model profiles, and GPU results were
removed on 13 September 2026. Chapter 2 now implements the revised service and
measurement story. The run log below preserves historical accounting; its
approvals do not authorize new runs. The detailed plan remains in ignored
`author/Chapter_02_Plan.md`. The current-source November L40S pair and historical
September L4 pair are separate comparisons in the Chapter 2 artifacts.

## Modal is a replaceable GPU runner

Prioritize clear teaching, representative engine behavior and transferable
engineering practices. Proceed with implementation, necessary diagnostics and
chapter experiments within the requested scope. Record each run's purpose,
configuration, evidence and outcome.

Choose examples, hardware, sample sizes and engine settings for clear teaching
and representative behavior. Do not disable normal engine features, truncate
necessary experiments or choose artificial defaults solely to minimize a
cloud bill. A small model remains appropriate when it isolates the chapter's
question and lowers reader prerequisites. Keep larger models in the chapters
where they explain a new mechanism; spending freedom is not a reason for an
unnecessary full benchmark matrix.

Modal is a replaceable compute adapter. Engine launch, readiness, request
handling, measurement and analysis should be executable on an ordinary Linux
GPU machine or in an engine container, using the same configuration and code.
Put provider-specific provisioning, credentials and artifact transfer only in
the adapter. Clearly distinguish client, API server, engine and framework
responsibilities and document the actual measurement topology.

Keep bounded concurrency, request/run deadlines, cancellation, failure records,
partial evidence and verified cleanup as operational teaching patterns. Allow
realistic startup and warmup for normal engine execution. Avoid hidden retries
in measurements; make diagnostic reruns explicit and preserve their evidence.
Record resources, versions and observed spend when available, but distinguish
measurement duration from billing and estimates from actual charges.

## Modal run log

Record the configuration and outcome of each remote run.

| Date | Current spend | GPU | Timeout |
| --- | ---: | --- | ---: |
| 2026-07-22 | $0.00 reported | L4 | 600 s |
| 2026-07-22 | < $0.05 estimated; report lag | L4 | 600 s |
| 2026-07-27 | $0.02035 reported | A100-80GB | 3,600 s |
| 2026-08-23 | Not CLI-verifiable | L4 | 1,200 s |
| 2026-08-23 | First attempt aborted before GPU; not CLI-verifiable | L4 | 1,200 s |
| 2026-08-23 | Two attempts stopped before GPU; not CLI-verifiable | L4 | 1,200 s |
| 2026-08-23 | About $0.06 estimated GPU runtime; not CLI-verifiable | L4 | 1,200 s |
| 2026-08-23 | $0.10 reported | L4/H100 representative wave | 480/600/1,200 s GPU; 7,200 s CPU prep |
| 2026-08-23 | Initial representative attempt failed during CPU import; no GPU allocated | None | < 2 min before stop |
| 2026-08-23 | Corrected retry reached a healthy server, then failed during exact-length prompt construction | L4 | About 4.5 min |
| 2026-08-23 | Calibration completed, but excluded because vLLM default prefix caching remained enabled | L4 | About 4.5 min |
| 2026-08-23 | Explicit-disable run stopped after the project-local Modal client disconnected; no artifacts returned | L4 | Less than the 480 s bound |
| 2026-08-23 | Clean 1.5B BF16 calibration completed; three bundles checksum-validated | L4 | Less than the 480 s bound |
| 2026-08-23 | 122B BF16/FP8 checkpoint preparation approved in parallel; GPU launch remains gated on 27B Qwen3.5 smoke | None for preparation | 7,200 s per preparation |
| 2026-08-23 | 7B BF16 and AWQ calibrations completed in parallel; six bundles checksum-validated | 2 x L4 | 600 s per GPU function |
| 2026-08-23 | 27B BF16 completed and validated; 27B FP8 reached DeepGEMM warmup but exceeded the generic 600 s readiness cap | 2 x H100 | 1,200 s per GPU function |
| 2026-08-23 | 27B Qwen3.5 gate satisfied; cached 122B BF16/FP8 GPU pair released | 3 x H200 total | 2,400 s BF16; 1,800 s FP8 |
| 2026-08-23 | Cached 27B FP8 retry completed; three bundles checksum-validated | H100 | Less than 1,200 s |
| 2026-08-23 | 122B BF16 completed and validated; 122B FP8 saved its first-time compile artifacts but exceeded the 1,350 s readiness cap during remaining warmup | 2 x H200 BF16; 1 x H200 FP8 | Less than 2,400/1,800 s bounds |

The final 2026-08-23 retry completed successfully. Based on observed runtimes
and published per-second rates, the two GPU-starting attempts plus image-build
CPU time are conservatively estimated below $0.20 total. The dashboard remains
the source of truth because the CLI cannot report billed usage.

## Keeping Python dependencies local on macOS

On 13 September 2026, imports and executables repeatedly stalled because the
project's `.venv` under Documents contained more than 26,000 macOS `dataless`
files. Their contents had been offloaded by the file provider. Stack traces
stopped in file reads during imports; this was not evidence of an incompatible
PyTorch/Transformers pair. A fresh install from the unchanged lockfile outside
Documents completed successfully.

The working repository now lives at `~/Developer/llm-inference-engineering`,
outside iCloud-managed Desktop and Documents. Keep future checkouts outside
cloud-managed folders too; a local virtual environment alone does not prevent
source or Git-object reads from waiting on cloud downloads.

On this laptop, `.venv` is now a directory symlink to
`~/.local/share/venvs/llm-inference-engineering`. The environment, uv's package
cache, and uv's managed Python stay outside Documents. Normal commands and
editor interpreter selection still use `.venv/bin/python`:

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```

For a fresh macOS checkout, use:

```bash
sh scripts/setup_macos.sh
```

The helper creates a dedicated environment for this book from the lockfile and
links `.venv` to it. It refuses to overwrite an existing directory or a link to
another environment. Do not share this environment between different projects
or simultaneously use it for checkouts with different lockfiles. Other projects
should use their own environment name. Linux CI keeps its normal local `.venv`.

If imports become slow, run this standard-library diagnostic without invoking
`uv run` first:

```bash
python3 scripts/check_python.py
```

It scans dependency metadata for offloaded files before importing packages,
reports disk space, and bounds the import check with a timeout. It never deletes
files or changes settings. Run it with `--environment /path/to/another/.venv` to
inspect another project. An offloaded environment is disposable: preserve any
manual package choices, rebuild from its project's lockfile outside the synced
folder, and check the replacement before deleting the old environment.

The laptop had about 4.6 GiB available during diagnosis. Keep adequate space for
package downloads and extraction. Avoid storing virtual environments, package
caches, and model weights in Desktop/Documents when a cloud provider manages
those folders. Keep active source checkouts downloaded too; Git's object files
can also be offloaded. Finder's **Keep Downloaded** is appropriate for source
folders that must remain in iCloud Drive. Do not disable machine-wide sync or
remove personal files as part of a dependency repair.

References: [Apple on keeping files downloaded](https://support.apple.com/en-gb/guide/mac-help/mchl1a02d711/mac)
and [uv project environment configuration](https://docs.astral.sh/uv/concepts/projects/config/).


## Chapter 2 implementation — 13 September 2026

The author approved code implementation. The HTTP client, fixed experiment,
JSONL evidence, offline analysis and bounded server lifecycle are implemented.
`httpx==0.28.1` is now an exact direct dependency in the existing uv lock. GPU
engine packages remain confined to the pinned remote image.

Validation: 35 repository tests pass; Ruff passes. A local synthetic HTTP run
completed all 171 scheduled requests, peaked at four concurrent connections,
and reconstructed the saved streams into six repeat/condition summaries. Its
bundle is ignored under `artifacts/tmp/chapter02-local-validation`. These are
fixture observations, not GPU or model-performance evidence.

The [run specification](chapters/02-building-a-measurable-llm-service/RUN_SPEC.md)
records the current setup. Initial app `ap-sPA65WIc1VxmoPqOnfJCzy` stopped
before Function registration, with zero tasks confirmed. Its console output
is `artifacts/tmp/chapter02-vllm-first-console.log`. Manuscript drafting and
final figures follow validated remote evidence.

### First attempt outcome and prepared correction

At 21:08 PDT on 13 September 2026 the first attempt stopped after image
preparation. Modal could not determine the image's Python version while
registering `prepare_weights`. No model download, generation request or GPU
Function execution occurred. The console log records four successful image
build stages (144.14, 4.16, 4.80 and 5.34 seconds). This is billable-stage
evidence, not a dashboard cost; the actual charge remains unverified.

Diagnosis: the pinned vLLM Dockerfile installs `python3` but does not create
the unversioned `python` command required by Modal Functions. The wrapper now
adds that alias only if missing and prints interpreter/pip versions during
build. It retains the image's existing Python and engine packages. This
correction passed the subsequent remote build and Function registration.

The corrected attempt used fresh output `artifacts/tmp/chapter02-vllm-second`;
its console log is `artifacts/tmp/chapter02-vllm-second-console.log`.

### Completed vLLM run

Corrected attempt `ap-7Zaxd3QFpeNn4SFjvapUWb`, run
`vllm-20260914T041032Z`, completed: 144/144 measured successes, 24 warmups,
two smoke checks and one deliberate client cancellation. Modal reported
stopped with zero tasks. The client experiment took 40.83 seconds; full
startup/preparation time and the actual dashboard bill are separate.

Prompt-count review caught a Transformers 5 return-type assumption. The raw
manifest is preserved, with a separate correction: cached-tokenizer
retokenization matched every measured server count (37–44 tokens). The helper
now explicitly requests token IDs and has a local regression test.

Prefer representative defaults and portable examples. The completed eager-mode
run is preliminary evidence, not the final comparison baseline.

Representative baseline now runs through platform-neutral `service.py`, also
used by the Modal adapter. Normal engine execution is enabled (no forced eager
flag), scheduler limit is 64, GPU function has four CPU cores and a 900-second
readiness allowance. The revised settings are recorded separately from the
eager-mode evidence; this is not an isolated CUDA-graph A/B experiment.

The standard SGLang run `sglang-20260914T043435Z` also completed 144/144
measured requests. Both initial standard apps stopped with zero tasks. The
first SGLang concurrent repeat was slower than the next two, and the warmup
covered only short uppercase tasks. A revised matched pair now warms up with
all 24 prompts per condition (291 total requests, 144 measured) and signals
the server parent first during shutdown. Local tests include cooperative
worker drainage. Earlier observations are retained as diagnostic evidence.

Revised pair (both now stopped): vLLM app `ap-dJ6sl3cU1fJO08WykBUC3G`, SGLang app
`ap-O3whEBVpHrn3Q2pGb3HAkA`. Both use one L4 and four CPU cores, independently.


The full-warmup vLLM run `vllm-20260914T044208Z` completed. SGLang attempt
`sglang-20260914T044205Z` stopped at its first smoke request: `/v1/models`
responded while internal server warmup was still running. The failed record
and server log are retained. The correction waits on `/health` with SGLang
health generation disabled; the measured timeout settings are unchanged.
A local readiness-race test passes. Retry `sglang-20260914T044837Z` uses app
`ap-m5JwRWXuftDdmszTAa3b4P` and the current recommended `sglang serve` command.


### Final Chapter 2 pair validated

SGLang retry `sglang-20260914T044837Z` completed after the readiness correction.
Together with vLLM `vllm-20260914T044208Z`, it provides 288/288 successful measured
requests, 288 successful warmups, four successful smoke checks and two deliberate
client cancellations. Prompt-count checks and event reconstruction passed;
executed source snapshots match their recorded hashes. All Modal apps were
verified stopped with zero tasks after the final run. Model and evidence
volumes remain available for reuse.

The [curated comparison](chapters/02-building-a-measurable-llm-service/artifacts/comparison-20260914/README.md)
contains per-repeat results, reproducible analysis, controls and limitations.
The failed readiness attempt remains separate diagnostic evidence. Engine
shutdown warnings are retained; stopped app state does not establish per-request
resource reclamation for the client cancellation probes.


## Chapter 2 current-source verification — 16 November 2026 UTC

The author requested implementation, remote validation and a PR. The code review
loop converged with two clean rounds (eleven and twelve). The full CPU suite
passed 83 tests and Ruff passed. Reviewed source commit:
`2f4a4972bf7ab866fdf7f56ae09bc3f530911b37`.

| Run | Hardware | Outcome |
| --- | --- | --- |
| vllm-20261116T061927Z | L4 requested | CPU preparation completed; GPU call stayed queued; explicitly stopped with no measurements. |
| vllm-20261116T064504Z | L4 requested | CPU preparation completed; GPU call stayed queued; explicitly stopped; bounded adapter retained failure/cancellation records. |
| vllm-20261116T065829Z | One L40S | Completed: 144/144 measured successes, 291 terminal records. |
| sglang-20261116T070238Z | One L40S | Completed: 144/144 measured successes, 291 terminal records. |

The L40S pair keeps the pinned model/images, BF16, four CPU cores, 16 GiB RAM,
full warmup, workload and deadlines. Both actual device records identify L40S.
Both metrics snapshots returned HTTP 200 in each run. Current executed-source
hashes, events, strict outputs, prompt counts and schedules passed validation.
All four apps were observed stopped with zero tasks and no live containers.
Actual billed spend was unavailable; experiment duration is not billing duration.

The adapter now bounds submission and result waiting, including scheduling,
and requests bounded cancellation on failure. Cleanup acknowledgment remains
separate from observed stopped state. Model/evidence volumes remain available.
The [current comparison](chapters/02-building-a-measurable-llm-service/artifacts/comparison-20261116/README.md)
is separate from historical L4 evidence. Local provisioning IDs and unnecessary
console logs remain ignored. Per-request engine cancellation remains unverified.
