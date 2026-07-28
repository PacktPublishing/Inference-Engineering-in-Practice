# Author development workflow

This file describes the book-development workflow. It is not manuscript
content and its spending limit is not guidance for readers.

## Git ownership

The author owns repository history. Agents may edit and validate the working
tree, but must never commit or push. The author reviews, commits, and pushes all
changes.

## Bootstrap

Use Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install modal
modal setup
uv sync
```

`modal setup` authenticates the local CLI. It does not run GPU code.

## Modal is a GPU sandbox only

Modal is used only when a result genuinely requires an NVIDIA GPU. The
educational modules stay platform-neutral; `modal_app.py` is a thin remote
runner around them.

Before every chargeable Modal command:

1. Run all local tests first.
2. Check current-month Modal usage and remaining credits with `modal billing`
   or the Modal dashboard.
3. Estimate the worst-case run cost from the selected GPU's current per-second
   price and the function timeout.
4. Record the date, current spend, remaining credit, GPU, timeout, and estimate
   in the run log below.
5. Ask the author before starting any remote code execution.

The hard development limit is **$30 of total Modal spend per calendar month**.
Do not start a job when current spend plus the worst-case estimate could exceed
that limit. Ask the author whether to reduce the experiment or explicitly
change the limit.

Use one L4, one container, bounded timeouts, zero warm containers, and no
deployed endpoint. Consolidate GPU experiments into one run after local
correctness checks pass. A persistent Volume may cache downloaded weights, but
no compute should remain active after the function returns.

## Modal run log

Add one row before each approved remote run.

| Date | Current spend | Credit left | GPU | Timeout | Worst-case estimate | Approved |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 2026-07-22 | $0.00 reported | $30.00 internal cap remaining | L4 | 600 s | < $0.25 | Yes, in chat |
| 2026-07-22 | < $0.05 estimated; report lag | > $29.95 internal cap remaining | L4 | 600 s | < $0.25 | Yes, corrected run in chat |
| 2026-07-27 | $0.02035 reported | $29.98 internal cap remaining | A100-80GB | 3,600 s | < $2.75 | Yes, cat edit in chat |
