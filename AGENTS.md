# Repository agent instructions

This repository contains runnable companion code for *Inference Engineering in
Practice*, by Dewang Sultania. The current checkout belongs outside iCloud:
`~/Developer/llm-inference-engineering`. Read `DEVELOPMENT.md` for environment
setup and historical run accounting. Author-only plans and manuscripts remain
local and ignored by Git.

These instructions adapt the principles in
[multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills)
(MIT licensed) to this project. This file is the single repository agent guide.

## Working principles

1. **Think before coding.** Read the relevant plan, source, tests and artifact
   notes. State material assumptions and define observable success. Resolve
   routine implementation choices within the author's requested scope.
2. **Prefer the simplest sufficient change.** Add code when a chapter needs it.
   Introduce shared infrastructure when actual reuse justifies it. Do not build
   speculative engine registries, model matrices or benchmark frameworks.
   Design for clear teaching, correctness, testing and portability. Write new
   code or replace existing code when that produces a better example. Existing
   code is a reference, not a constraint; preserve historical evidence and its
   executed source when replacing an implementation.
3. **Make surgical edits.** Every change should support the requested outcome.
   Preserve unrelated work and historical evidence unless its removal is part
   of the author's request.
4. **Verify what matters.** Use focused checks for behavior and meaningful
   failures; do not add tests that merely mirror implementation. Report missing
   validation and incomplete evidence plainly.
5. **Respect author ownership.** Commit or push only when explicitly requested
   for the current work. Never send editor correspondence or submit a manuscript
   without the author's instruction to do so.

## Teaching and writing principles

The audience is software engineers operating LLM services. Organize chapters
around concepts, failure investigations and engineering decisions. Teach a
concept once, then compare vLLM, SGLang and TensorRT-LLM where their differences
help explain behavior. Keep hardware and kernels as supporting context.

Distinguish the engine (generation and request state), API server (request and
response contract), and serving framework (routing and replica lifecycle).
API handling and framework coordination are parts of the broad serving layer;
they need not be separate processes. Define the selected replica's scope,
separate data-path arrows from control actions, and map settings to their owner.
Follow one request, then one replica, then a fleet. Engine comparisons and
framework comparisons change different variables; a packaged API frontend can
affect an engine comparison. Use the author notebook review when available,
but independently verify its claims and create original diagrams and evidence.

In end-to-end lessons, establish the three-component map once and preserve its
labels across examples. Revisit it only when a new boundary, scope or failure
needs explanation. Prefer a brief reference or highlighted detail to repeated
definitions, full diagrams and recaps. Pair snippets with purpose and actual
output where useful; each example or figure should add understanding. Keep
conceptual framework illustrations distinct from components executed in the lab.

For Chapter 2, implement the code, make the complete flow work, pass relevant
checks, collect validated remote results and validate the evidence before writing
the chapter prose or finalizing manuscript figures and excerpts. A storyboard
can guide implementation, but it is not a substitute for working code. Apply
Mark's feedback during drafting and remove repetition as well as logical gaps.

Apply Mark's editorial feedback throughout, including uncommented passages:

- Establish the reader's question and why the next idea matters before
  introducing it. Connect each section to what readers already understand.
- Define unfamiliar terms at their first meaningful use. Use concrete examples
  and explain symbols and units needed for a calculation.
- Make causal links explicit. Explain how an observation supports a conclusion,
  and distinguish evidence from a possible explanation.
- Replace vague subjects, incomplete statements and unsupported generalizations
  with specific actors, actions, conditions and limits.
- Write transitions between sentences, paragraphs and sections. Recall earlier
  concepts briefly without repeating whole lessons.
- Before code, state its purpose, inputs and what to look for. After code,
  interpret its output and connect it to the next decision.

Plan figures before drafting prose: request paths, state changes, timing
boundaries and comparisons should be visible. Each figure answers a question;
its caption explains the takeaway and any important boundary. Label illustrative
values separately from measured results. Use short essential excerpts in prose
and keep complete runnable programs in the repository. Do not force a repeated
lesson template into every section.

Aim for roughly 6–10 purposeful conceptual diagrams per chapter, plus measured
plots and short actual output excerpts where experiments are taught. Introduce
the question, show the diagram/code/output, then explain what the reader can
conclude. Preserve run IDs for excerpts; mark elisions and fixture data. Do not
invent successful output or leave all results in repository logs. Plan model
sizes across chapters according to `BOOK_PLAN.md`; avoid running every size on
every engine merely for coverage. Chapter 2 introduces a small measured engine
comparison; later chapters deepen it around the mechanism being taught.

For reviewed Word manuscripts, preserve existing revisions, Mark's comments and
prior replies. Track new edits and attribute edits and comment replies to
**Dewang Sultania**. Reply directly to every comment addressed, including cuts
or decisions to retain text; do not merely resolve comments. Render and inspect
the document before delivery. The author reviews and submits the final file.

## Repository and evidence guardrails

- Never stage or commit `BOOK_PLAN.md`, `author/`, `chapters/**/chapter.md`,
  publisher documents, PDFs, credentials, model weights or `artifacts/tmp/`.
  Preserve ignored author files during cleanup and migration.
- Preserve the Packt repository history and `LICENSE`.
- Keep chapter programs in `code/` and focused tests in `tests/`. Manuscript
  excerpts must agree with the runnable source.
- Keep inference/client code platform-neutral and Modal integration thin.
  Keep engine image dependencies separate from the laptop environment.
- Pin model revisions and tested dependency versions. Verify the chosen
  engine release's supported model, GPU, API and launch controls before running.
- Keep raw outcomes, failures, timing boundaries, token-count provenance and
  effective configuration. Qualify claims by workload, versions and hardware.
  A stream chunk is not a token; a closed socket alone does not prove that
  engine work stopped. Never fabricate a result or treat an unrun engine as slow.
- Curated artifacts need reproduction commands, metadata and clear limitations.
  Keep private notes, credentials and unnecessary logs out of public artifacts.
- CI stays CPU-only, credential-free and independent of model downloads.

## Remote execution and teaching priorities

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

## Repeated independent review

Use the [Elephant–Goldfish method](https://github.com/google/elephant-goldfish-model)
for code and chapter review. The primary agent retains project context and
resolves findings. Goldfish reviewers must be fresh subagents, using
`fork_turns="none"` where available, with declared artifacts, standards and
source material rather than the drafting conversation or private rationale.
Keep reviewers read-only and wait for their actual reports before making edits.

Review code for teaching clarity, correctness, meaningful tests, failure handling,
portability and reproducible evidence. Review the full manuscript for Mark's
criteria: context, definitions, causal links, precise statements, transitions,
purpose before examples, interpretation after output, and removal of repetition.
Check component boundaries, figures, excerpts and measured claims against source.

Repeat review, justified repair and affected validation within the authorized
scope. Require two consecutive clean rounds with all reviewers completed and
no unresolved major or blocking finding. Judge a round before applying repairs;
material edits reset the clean-round count. Failed or missing reviews do not
count as clean. Reopen findings when a supposed fix fails. Keep checkable reasons
for contested findings; do not dismiss reader comprehension gaps as intentional.

Use five rounds as an initial limit. At the limit, or when repairs repeatedly
reverse earlier repairs, report open issues and ask for direction instead of
claiming completion. Required technical checks and claim evidence cannot be
silently waived. Keep inputs, findings and decisions in ignored `author/reviews/`.
Code changes during manuscript review return to code and evidence validation.
The book plan defines the detailed review configuration and completion checks.

## Standard workflow

1. Read the relevant chapter plan and define the smallest experiment and its
   success checks. For Chapter 2, use `author/Chapter_02_Plan.md` when available.
2. Implement the required local pieces and test streaming, failure accounting
   and arithmetic before renting a GPU.
3. Freeze deployment inputs and document the engineering purpose, configuration,
   resources, deadlines and actual client/server topology before running.
4. Collect and validate results. Analyze locally and trace manuscript claims
   and plots to saved evidence. Narrow claims when evidence is incomplete.
5. Draft figures and prose, then review reader context, definitions, transitions,
   example purpose, technical accuracy and manuscript layout.
6. Run checks appropriate to the change, inspect the final diff and confirm
   private author files remain ignored. Report what changed and what was verified.

## Local validation

For code changes, use the narrowest useful checks first, then relevant project
checks. Documentation-only changes need link and diff checks rather than new
behavior tests. See `DEVELOPMENT.md` for the macOS environment repair procedure.

```bash
uv sync --locked
uv run ruff check .
uv run pytest
```
