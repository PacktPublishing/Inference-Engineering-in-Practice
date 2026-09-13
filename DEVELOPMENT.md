# Author development workflow

This file describes the book-development workflow. It is not manuscript
content and its spending limit is not guidance for readers.

## Git ownership

The author owns repository history. Agents may edit and validate the working
tree, but must never commit or push. The author reviews, commits, and pushes all
changes.

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

Chapter 1 is the only implemented chapter. The former Chapter 2 implementation,
shared serving benchmarks, simulations, model profiles, and GPU results were
removed on 13 September 2026. Rebuild Chapter 2 from its revised service and
measurement story, adding code only as each example needs it. The run log below
is historical accounting; its approvals do not authorize new runs.

## Modal is a GPU sandbox only

Modal is used only when a result genuinely requires an NVIDIA GPU. The
educational modules stay platform-neutral; `modal_app.py` is a thin remote
runner around them.

Before every chargeable Modal command:

1. Run all local tests first.
2. Check current-month Modal usage and remaining credits in the Modal dashboard.
   Modal CLI 1.2.6 does not provide a `modal billing` command.
3. Estimate the worst-case run cost from the selected GPU's current per-second
   price and the function timeout.
4. Record the date, current spend, remaining credit, GPU, timeout, and estimate
   in the run log below.
5. Ask the author before starting any remote code execution.

The hard development limit is **$30 of total Modal spend per calendar month**.
Do not start a job when current spend plus the worst-case estimate could exceed
that limit. Ask the author whether to reduce the experiment or explicitly
change the limit.

Use the least expensive practical GPU type/count pinned by the selected model
profile, one container, bounded timeouts, zero warm containers, zero retries,
and no deployed endpoint. Prepare immutable model weights in a separate
CPU-only function before measured GPU work. Consolidate related experiments
after local correctness checks pass. A persistent Volume may cache downloaded
weights, but no compute should remain active after the function returns.

## Modal run log

Add one row before each approved remote run.

| Date | Current spend | Credit left | GPU | Timeout | Worst-case estimate | Approved |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 2026-07-22 | $0.00 reported | $30.00 internal cap remaining | L4 | 600 s | < $0.25 | Yes, in chat |
| 2026-07-22 | < $0.05 estimated; report lag | > $29.95 internal cap remaining | L4 | 600 s | < $0.25 | Yes, corrected run in chat |
| 2026-07-27 | $0.02035 reported | $29.98 internal cap remaining | A100-80GB | 3,600 s | < $2.75 | Yes, cat edit in chat |
| 2026-08-23 | Not CLI-verifiable | $30 credit reported available | L4 | 1,200 s | < $0.75 | Yes, exact bounded run in chat |
| 2026-08-23 | First attempt aborted before GPU; not CLI-verifiable | About $30 reported available | L4 | 1,200 s | < $0.75 | Yes, corrected CUDA 12.9 retry in chat |
| 2026-08-23 | Two attempts stopped before GPU; not CLI-verifiable | About $30 reported available | L4 | 1,200 s | < $0.75 | Yes, dependency-fixed retry in chat |
| 2026-08-23 | About $0.06 estimated GPU runtime; not CLI-verifiable | About $29.94 estimated available | L4 | 1,200 s | < $0.75 | Yes, usage-chunk parser retry in chat |
| 2026-08-23 | $0.10 reported | $29.90 internal cap remaining | L4/H100 representative wave | 480/600/1,200 s GPU; 7,200 s CPU prep | $7.6243 maximum including contingency | Yes, exact small/mid wave in chat |
| 2026-08-23 | Initial representative attempt failed during CPU import; no GPU allocated | At least $29.85 estimated internal cap remaining | None | < 2 min before stop | < $0.05 estimated, covered by contingency | Yes, corrected retry in chat |
| 2026-08-23 | Corrected retry reached a healthy server, then failed during exact-length prompt construction | At least $29.76 estimated internal cap remaining | L4 | About 4.5 min | < $0.09 estimated, covered by contingency | Yes, tokenizer-fixed retry in chat |
| 2026-08-23 | Calibration completed, but excluded because vLLM default prefix caching remained enabled | At least $29.67 estimated internal cap remaining | L4 | About 4.5 min | < $0.09 estimated, covered by contingency | Yes, explicit-disable retry in chat |
| 2026-08-23 | Explicit-disable run stopped after the project-local Modal client disconnected; no artifacts returned | At least $29.52 estimated internal cap remaining | L4 | Less than the 480 s bound | < $0.1488 maximum | Yes, clean-client retry in chat |
| 2026-08-23 | Clean 1.5B BF16 calibration completed; three bundles checksum-validated | At least $29.37 estimated internal cap remaining | L4 | Less than the 480 s bound | < $0.1488 maximum | Yes, approved representative wave |
| 2026-08-23 | 122B BF16/FP8 checkpoint preparation approved in parallel; GPU launch remains gated on 27B Qwen3.5 smoke | At least $29.37 before new operations | None for preparation | 7,200 s per preparation | $1.266048 combined preparation maximum; $13.505848 full wave maximum | Yes, staged parallel schedule in chat |
| 2026-08-23 | 7B BF16 and AWQ calibrations completed in parallel; six bundles checksum-validated | At least $27.69 before large-model preparation | 2 x L4 | 600 s per GPU function | $1.680576 combined preparation and run maximum | Yes, staged parallel schedule in chat |
| 2026-08-23 | 27B BF16 completed and validated; 27B FP8 reached DeepGEMM warmup but exceeded the generic 600 s readiness cap | At least $21.76 after large preparation and original 27B maximum exposures | 2 x H100 | 1,200 s per GPU function | $4.661856 combined preparation and run maximum | Yes, cached proportional-readiness FP8 retry in chat |
| 2026-08-23 | 27B Qwen3.5 gate satisfied; cached 122B BF16/FP8 GPU pair released | At least $20.06 before 122B GPU maximum exposure | 3 x H200 total | 2,400 s BF16; 1,800 s FP8 | $12.2398 remaining GPU operations plus contingency | Yes, staged parallel schedule in chat |
| 2026-08-23 | Cached 27B FP8 retry completed; three bundles checksum-validated | Covered by the approved representative-wave maximum | H100 | Less than 1,200 s | No increase beyond approved timeout exposure | Yes, cached proportional-readiness FP8 retry in chat |
| 2026-08-23 | 122B BF16 completed and validated; 122B FP8 saved its first-time compile artifacts but exceeded the 1,350 s readiness cap during remaining warmup | Covered by the approved 122B wave maximum | 2 x H200 BF16; 1 x H200 FP8 | Less than 2,400/1,800 s bounds | No increase beyond approved wave exposure; cached FP8 retry requires fresh approval | Yes, staged parallel schedule in chat |

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
