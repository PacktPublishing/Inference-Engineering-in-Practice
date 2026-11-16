# Evidence bundles

Start with the [validated matched comparison](comparison-20260914/README.md).
The final [vLLM](vllm-20260914T044208Z/README.md) and
[SGLang](sglang-20260914T044837Z/README.md) bundles each contain 144 successful
measurements, full-workload warmup, three repeats and exact executed source.
All Modal apps were verified stopped with zero tasks.

Earlier runs remain diagnostic evidence: the
[eager vLLM run](vllm-20260914T041032Z/README.md) includes a documented prompt-count
metadata correction; the short-warmup
[vLLM](vllm-20260914T042253Z/README.md) and
[SGLang](sglang-20260914T043435Z/README.md) runs motivated representative warmup;
the [readiness race](diagnostics/readiness-race/README.md) explains the health
check correction. Development fixtures are synthetic and are not model performance.

Keep temporary bundles under the root `artifacts/tmp/` directory, ignored by
Git. Promote only reviewed, reproducible evidence into this chapter directory.
For every curated bundle include the reproduction command, observed cost,
run ID, source hashes, model/image revisions, exact hardware and limits,
completed/incomplete status, and a short account of what the run can establish.

- `launch.json`: selected resource and server controls before startup.
- `server.log`, `metrics-before.txt`, `metrics-after.txt`: supporting engine
  evidence; an absent snapshot is not a zero measurement.
- `outcome.json`: overall remote outcome, including startup failures.
- `run/manifest.json`: workload, topology, model preparation, package versions,
  effective prefix-cache evidence, planned IDs and terminal count.
- `run/requests.jsonl`: every started request's terminal outcome, including
  smoke, warmup, failures and the deliberate cancellation probe.
- `run/events.jsonl`: submitted payloads, protocol events and terminal status.
- `run/summary.csv`, `run/measurements.png`: locally regenerated analysis.

Do not merge warmup or probes into measurements. Preserve incomplete bundles
and missing token counts. Report individual repeats and actual output lengths;
these small samples do not establish a stable tail latency or engine ranking.
