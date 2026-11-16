# Metadata availability was mistaken for inference readiness

SGLang run `sglang-20260914T044205Z` stopped at its first non-streaming smoke
request. The API returned `/v1/models` at 04:45:10 UTC, but internal warmup did
not finish until 04:45:21. The client’s ten-second idle timer expired first.
No measured requests ran. The failure is retained as an incomplete run.

The corrected service waits for `/health` with
`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0`. In this selected release that
endpoint returns 503 while the server is Starting and does not generate health
traffic once ready. Only then do contract checks and representative workload
warmup begin. The measured request deadlines were not relaxed.

`server.log`, `run/requests.jsonl`, `run/events.jsonl`, `run/manifest.json` and
`outcome.json` preserve the failure. `source/` contains the executed code before
the readiness correction. App `ap-O3whEBVpHrn3Q2pGb3HAkA` stopped with zero
tasks. The parent-first shutdown reached zero remaining requests; the selected
engine logged a shutdown traceback and NCCL cleanup warning, which remain in
the raw log.
