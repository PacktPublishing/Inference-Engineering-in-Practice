# Repository agent instructions

This repository contains runnable companion code for *Inference Engineering in
Practice*. The publisher manuscript is maintained separately.

These instructions adapt the principles in
[multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)
(MIT licensed) to this project.

## Working principles

1. **Think before coding.** State material assumptions, surface ambiguity, and
   ask before making a choice that changes scope, cost, or published behavior.
2. **Prefer the simplest sufficient change.** Do not add speculative
   abstractions, configurability, or defensive code for impossible cases.
3. **Make surgical edits.** Every changed line should support the requested
   outcome. Do not refactor, reformat, or remove unrelated code.
4. **Work toward verifiable goals.** Define success checks before editing and
   iterate until they pass. Report incomplete validation honestly.
5. **Leave version control to the author.** Never create commits or push
   branches. Make and validate working-tree changes, then ask the user to
   review, commit, and push them.

## Repository guardrails

- Never stage or commit `chapters/**/chapter.md`, publisher documents, PDFs,
  credentials, model weights, or files under `artifacts/tmp/`.
- Preserve the Packt repository history and `LICENSE`.
- Keep complete runnable programs in each chapter's `code/` directory and
  focused tests in `tests/`. Chapter prose should contain only essential
  excerpts and must agree with the source code.
- Prefer platform-neutral inference code. Keep Modal integration in a thin
  wrapper rather than coupling examples to Modal.
- Use pinned model revisions and locked dependency versions. Qualify claims
  whose results depend on hardware, software versions, model architecture, or
  workload shape.
- Curated artifacts must include enough metadata to reproduce or interpret
  them. Update their README when regeneration steps or provenance change.
- CI must remain CPU-only, credential-free, and independent of model downloads.

## Modal and cost safety

- Develop and test locally first.
- Do not start any Modal execution without explicit user approval for that run.
- Before requesting approval, state the GPU, timeout, expected workload, and a
  conservative cost estimate. Respect the documented $30 monthly cap.
- Record approved remote runs in `DEVELOPMENT.md`; do not expose secrets or
  credentials in logs or artifacts.

## Standard workflow

1. Read the relevant chapter README, source, tests, and artifact notes.
2. Define the smallest implementation and its success criteria.
3. Add or update focused tests, then implement the change.
4. Run the narrowest useful checks, followed by the full local checks when
   practical.
5. Inspect the final diff for unrelated edits and verify ignored manuscript
   files remain untracked.
6. Stop with validated working-tree changes and ask the user to review, commit,
   and push them.

## Validation

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```
