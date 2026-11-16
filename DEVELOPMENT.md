# Development

Use Python 3.12 and the checked-in lockfile:

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```

CI runs these CPU-only checks without credentials or model downloads. Chapter 1
uses a tiny local model in its tests. Chapter 2 uses synthetic HTTP responses to
check streaming, timing arithmetic, failure records, readiness, and cleanup.
These fixtures are not GPU measurements.

Keep programs in each chapter's `code/` directory and behavioral tests in its
`tests/` directory. The root `tests/` checks public documentation links. Keep
engine dependencies in separate GPU environments. The chapter READMEs describe
local analysis, Linux GPU execution, and optional Modal adapters.

Write new results to a fresh directory under `artifacts/tmp/`. Preserve raw
records locally when investigating a result. Commit only small datasets or
excerpts that support a reader exercise, with provenance, units, reproduction
commands, and limits. Do not copy source trees into every output directory;
record the source revision and retain local changes separately when needed.

Before a PR, inspect its file list. Exclude author plans, manuscripts, publisher
files, credentials, weights, raw runs, temporary plots, and private review notes.
Keep `LICENSE` and repository history. Author-only material stays ignored under
`author/`, `BOOK_PLAN.md`, and chapter `chapter.md` files.
