# Synthetic misleading-result demonstrations

Run identity: `fixtures-20261115`. These are executed CPU fixture results,
not LLM, GPU, or network performance. Token usage is invented by the fixture.

From the repository root:

```bash
uv run python chapters/02-building-a-measurable-llm-service/code/fixture_demo.py \
  --output artifacts/tmp/chapter02-fixtures
```

The client receives `hello` in one content event and then in five content events.
Both responses report one fixture-defined token. In the five-event case,
counting events as tokens gives 444.16 events/s instead of 88.83 fixture tokens/s.
The factor of five comes from the counting error, not a faster model.

The second demonstration records one success and one deadline failure.
Keeping both yields 8.78 successful requests/s with a 50% failure fraction.
Dropping the failed request yields the misleading 85.52 requests/s and 0% failures.
The denominator changed because the slower failed request disappeared.
These timings vary across local runs; the within-run arithmetic is the lesson.

`demonstrations.json` contains correct and explicitly wrong calculations.
Each case saves request/event records and labelled metadata. `source/` preserves
executed Python source, and `SHA256SUMS.json` records its generated file hashes.
This README was added after execution and is not in that execution checksum list.
