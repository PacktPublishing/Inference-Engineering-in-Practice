"""Reject evidence disagreement without a model, credentials, or GPU."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from client import ChatClient
from compare import read_run
from evidence import validate_events, validate_run
from experiment import run_experiment
from fixture_demo import demonstrate
from prepare_model import validate_preparation
from records import RequestSpec, content_hash
from service import capture_metrics
from test_service import endpoint, fixture_config

CHAPTER = Path(__file__).parents[1]
@pytest.fixture(scope="module")
def synthetic_runs(tmp_path_factory):
    """Exercise the real client and schedule against a small local HTTP fixture."""
    root = tmp_path_factory.mktemp("synthetic-evidence")
    config = fixture_config()
    config.update(repeats=2, concurrency=[1, 2], warmup_requests=2)
    config["prompts"] = config["prompts"][:2]
    for prompt in config["prompts"]:
        prompt["expected_text"] = "café"
    preparation = {
        "model": config["model"], "revision": config["model_revision"],
        "chat_template": "fixture", "chat_template_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "prompts_sha256": content_hash(config["prompts"]),
        "prompt_tokens": {p["id"]: 10 for p in config["prompts"]},
    }

    async def exercise():
        async with endpoint() as (url, _state):
            for engine in ("vllm", "sglang"):
                metadata = {
                    "engine": engine, "evidence_kind": "SYNTHETIC HTTP FIXTURE",
                    "resources": {"gpu": "synthetic"}, "topology": "local HTTP fixture",
                    "execution_mode": "fixture", "model_preparation": preparation,
                    "versions": {"fixture": "1"},
                }
                result = await run_experiment(url, config, root / engine, metadata)
                assert result["status"] == "completed", result["error"]
    asyncio.run(exercise())
    return root


def saved_run(root):
    directory = root / "vllm"
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = [json.loads(s) for s in (directory / "requests.jsonl").read_text().splitlines()]
    events = [json.loads(s) for s in (directory / "events.jsonl").read_text().splitlines()]
    return manifest, rows, events


def test_comparison_rejects_mixed_gpus_and_labels_selected_hardware(
    tmp_path, monkeypatch, synthetic_runs
):
    import compare as comparison
    from matplotlib.figure import Figure

    original_read = comparison.read_run
    selected = {"vllm": "L4:1", "sglang": "L40S:1"}

    def read_with_gpu(path):
        manifest, rows = original_read(path)
        # Test-only metadata variation; never written back to preserved evidence.
        manifest["metadata"]["resources"]["gpu"] = selected[manifest["metadata"]["engine"]]
        return manifest, rows

    monkeypatch.setattr(comparison, "read_run", read_with_gpu)
    other = synthetic_runs / "sglang"
    with pytest.raises(ValueError, match="Serving control differs: resources"):
        comparison.compare(synthetic_runs / "vllm", other, tmp_path / "mixed")
    assert not (tmp_path / "mixed").exists()
    selected["vllm"] = "L40S:1"
    titles = []
    ticks = []
    original_title = Figure.suptitle

    def capture_title(self, text, *args, **kwargs):
        titles.append(text)
        ticks.extend(axis.get_xticks().tolist() for axis in self.axes)
        return original_title(self, text, *args, **kwargs)

    monkeypatch.setattr(Figure, "suptitle", capture_title)
    comparison.compare(synthetic_runs / "vllm", other, tmp_path / "matched")
    assert titles == ["Serving comparison · L40S:1"]
    assert ticks == [[1, 2]] * 3
    assert (tmp_path / "matched/engine-comparison.png").is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_tokens", 1000000),
        ("prompt_tokens", 1),
        ("usage_source", "guessed"),
        ("last_content_s", 999),
        ("ended_s", 1000),
        ("finish_reason", "length"),
        ("text", "wrong"),
        ("first_content_s", 999),
    ],
)
def test_event_disagreement_is_rejected(field, value, synthetic_runs):
    _manifest, rows, events = saved_run(synthetic_runs)
    row = next(r for r in rows if r["phase"] == "measure")
    row[field] = value
    with pytest.raises(ValueError):
        validate_events(rows, events)


def test_ordinary_response_and_chronology_are_checked(synthetic_runs):
    _, rows, events = saved_run(synthetic_runs)
    rows[0]["text"] = "tampered ordinary response"
    with pytest.raises(ValueError, match="text"):
        validate_events(rows, events)
    _, rows, events = saved_run(synthetic_runs)
    events[1]["elapsed_s"] = float("nan")
    with pytest.raises(ValueError, match="chronology"):
        validate_events(rows, events)


@pytest.mark.parametrize("mutation", ["condition", "messages", "objective", "schedule", "hash"])
def test_run_rejects_same_count_but_wrong_evidence(mutation, synthetic_runs):
    manifest, rows, events = saved_run(synthetic_runs)
    validate_run(manifest, rows, events)
    if mutation == "condition":
        for row in rows:
            if row["phase"] == "measure":
                row["repeat"] = 0
    elif mutation == "messages":
        events[0]["payload"]["messages"][0]["content"] = "different prompt"
    elif mutation == "objective":
        rows[0]["output_check_passed"] = False
    elif mutation == "schedule":
        manifest["schedule"] = manifest["schedule"][:-1]
    else:
        manifest["config"]["max_tokens"] += 1
    with pytest.raises(ValueError):
        validate_run(manifest, rows, events)


def test_synthetic_comparison_inputs_pass_stronger_checks(synthetic_runs):
    for name in ("vllm", "sglang"):
        manifest, rows = read_run(synthetic_runs / name)
        assert manifest["generation_requests"] == len(rows) == 19


@pytest.mark.parametrize(
    "mutation", ["model", "revision", "prompt_ids", "prompt_content", "template"]
)
def test_prepared_model_must_match_run(mutation, synthetic_runs):
    manifest, _, _ = saved_run(synthetic_runs)
    config = manifest["config"]
    prep = copy.deepcopy(manifest["metadata"]["model_preparation"])
    prep["prompts_sha256"] = content_hash(config["prompts"])
    validate_preparation(config, prep)
    if mutation in ("model", "revision"):
        prep[mutation] = "other"
    elif mutation == "prompt_ids":
        prep["prompt_tokens"].pop(next(iter(prep["prompt_tokens"])))
    elif mutation == "prompt_content":
        config["prompts"][0]["messages"][0]["content"] = "changed"
    else:
        prep["chat_template"] += "changed"
    with pytest.raises(ValueError):
        validate_preparation(config, prep)


@pytest.mark.parametrize("text,passed", [("AMBER", True), ("amber", False), ("BIRCH", False)])
def test_exact_uppercase_objective(text, passed):
    async def exercise():
        def response(request):
            return httpx.Response(
                200,
                json={
                    "model": "fixture",
                    "id": "fixture",
                    "choices": [
                        {"index": 0, "message": {"content": text}, "finish_reason": "stop"}
                    ],
                },
            )

        async with ChatClient(
            "http://fixture/v1", "fixture", transport=httpx.MockTransport(response)
        ) as client:
            result = await client.generate(
                RequestSpec(
                    "r1",
                    "p1",
                    [],
                    stream=False,
                    expected_text="AMBER",
                ),
                origin=0,
            )
        assert result.status == "success"
        assert result.output_check_passed is passed

    asyncio.run(exercise())


@pytest.mark.parametrize("phase", ["before", "after"])
@pytest.mark.parametrize(
    "status,body,valid",
    [
        (500, "server error", False),
        (200, "not metrics", False),
        (200, "vllm:num_requests_running 0\nvllm:num_requests_waiting NaN\n", False),
        (
            200,
            'vllm:num_requests_running{model_name="fixture"} 0\nvllm:num_requests_waiting 1\n',
            True,
        ),
    ],
)
def test_metric_snapshots_validate_status_and_fields(tmp_path, phase, status, body, valid):
    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, text=body))
        ) as observer:
            if valid:
                assert await capture_metrics(observer, tmp_path, "vllm", phase) == body
            else:
                with pytest.raises((ValueError, httpx.HTTPStatusError)):
                    await capture_metrics(observer, tmp_path, "vllm", phase)
        record = json.loads((tmp_path / f"metrics-{phase}-observation.json").read_text())
        assert record["status"] == ("completed" if valid else "incomplete")
        assert (tmp_path / f"metrics-{phase}.txt").exists() == valid
        assert (tmp_path / f"metrics-{phase}-failed.txt").exists() != valid

    asyncio.run(exercise())


def test_saved_fixture_demonstrations(tmp_path):
    report = asyncio.run(demonstrate(tmp_path / "demo"))
    one, five = report["segmentation"]
    assert one["text"] == five["text"] == "hello"
    assert [one["content_events"], five["content_events"]] == [1, 5]
    assert one["fixture_reported_tokens"] == five["fixture_reported_tokens"] == 1
    assert five["wrong_events_as_tokens_per_s"] == pytest.approx(
        5 * five["correct_fixture_tokens_per_s"]
    )
    accounting = report["failure_accounting"]
    assert accounting["statuses"] == ["success", "deadline"]
    correct = accounting["correct_all_outcomes"]
    omitted = accounting["wrong_failed_request_removed"]
    assert correct["failure_fraction"] == 0.5
    assert omitted["failure_fraction"] == 0
    assert correct["requests_per_s"] < omitted["requests_per_s"]


def test_finished_workload_does_not_hide_failed_final_observation(
    tmp_path, monkeypatch, synthetic_runs
):
    from contextlib import asynccontextmanager

    import service

    manifest, _, _ = saved_run(synthetic_runs)
    config = manifest["config"]
    prepared = manifest["metadata"]["model_preparation"]
    prepared["prompts_sha256"] = content_hash(config["prompts"])
    model = tmp_path / "model"
    model.mkdir()
    (model / "preparation.json").write_text(json.dumps(prepared))
    state = {"stopped": False, "calls": 0}

    @asynccontextmanager
    async def server(*args, **kwargs):
        try:
            yield None
        finally:
            state["stopped"] = True

    async def experiment(*args, **kwargs):
        return {"status": "completed", "error": None}

    def response(request):
        state["calls"] += 1
        if state["calls"] == 2:
            return httpx.Response(500, text="failed final observation")
        return httpx.Response(
            200,
            text=(
                'vllm:cache_config_info{enable_prefix_caching="False"} 1\n'
                "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
            ),
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        service.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(response), **kwargs),
    )
    monkeypatch.setattr(service, "running_server", server)
    monkeypatch.setattr(service, "require_free_port", lambda: None)
    monkeypatch.setattr(service, "run_experiment", experiment)
    monkeypatch.setattr(service.importlib.metadata, "version", lambda name: "0.29.0")
    monkeypatch.setattr(service.subprocess, "check_output", lambda *a, **k: "synthetic GPU")
    result = asyncio.run(service.serve_and_measure(config, model, tmp_path / "run"))
    assert result["status"] == "incomplete"
    assert result["experiment_status"] == "completed"
    assert state["stopped"]
    assert "500" in result["error"]


def test_existing_listener_cannot_supply_service_readiness(tmp_path, monkeypatch, synthetic_runs):
    import socket
    from functools import partial

    import service

    manifest, _, _ = saved_run(synthetic_runs)
    with socket.socket() as existing:
        existing.bind(("127.0.0.1", 0))
        existing.listen()
        port = existing.getsockname()[1]
        monkeypatch.setattr(service, "require_free_port", partial(service.require_free_port, port))

        def unexpected_launch(*args, **kwargs):
            pytest.fail("An occupied port must stop the workflow before server launch")

        monkeypatch.setattr(service, "running_server", unexpected_launch)
        result = asyncio.run(
            service.serve_and_measure(manifest["config"], tmp_path / "model", tmp_path / "run")
        )
        assert result["status"] == "incomplete"
        assert result["experiment_status"] == "not_started"
        assert f"API port 127.0.0.1:{port} is unavailable" in result["error"]
        assert not (tmp_path / "run/server.log").exists()


@pytest.mark.parametrize("stream", [False, True])
def test_malformed_compression_is_persisted(tmp_path, stream):
    import time

    from experiment import run_condition
    from records import RunWriter

    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"not-a-gzip-stream"

    async def exercise():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
                stream=BrokenBody(),
            )
        )
        writer = RunWriter(tmp_path / "failure", {"status": "incomplete"})
        async with ChatClient("http://fixture/v1", "fixture", transport=transport) as client:
            results = await run_condition(
                client,
                [RequestSpec("r1", "p1", [], stream=stream)],
                writer,
                time.perf_counter(),
                1,
                time.perf_counter() + 60,
            )
        writer.close()
        assert results[0].status == "protocol_error"
        rows = [
            json.loads(s) for s in (tmp_path / "failure/requests.jsonl").read_text().splitlines()
        ]
        events = [
            json.loads(s) for s in (tmp_path / "failure/events.jsonl").read_text().splitlines()
        ]
        assert len(rows) == 1
        validate_events(rows, events)

    asyncio.run(exercise())


def test_cancel_records_combined_content_and_finish():
    import time
    from dataclasses import asdict

    async def exercise():
        payload = {
            "id": "fixture",
            "model": "fixture",
            "choices": [
                {"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"},
            ],
        }
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="data: " + json.dumps(payload) + "\n\n",
            )
        )
        async with ChatClient("http://fixture/v1", "fixture", transport=transport) as client:
            result = await client.generate(
                RequestSpec("r1", "p1", [], cancel_after_content=True), origin=time.perf_counter()
            )
        assert result.status == "cancelled"
        assert result.finish_reason == "stop"
        assert result.engine_cancellation == "unverified"
        validate_events(
            [asdict(result)], [{"request_id": result.request_id, **e} for e in result.events]
        )

    asyncio.run(exercise())


def test_linux_cli_metadata_supports_full_comparison(tmp_path, monkeypatch, synthetic_runs):
    """Exercise the documented CLI boundary without starting an engine or renting compute."""
    import shutil
    import sys

    import service
    from compare import compare

    config_path = tmp_path / "config.json"
    manifest, _, _ = saved_run(synthetic_runs)
    config_path.write_text(json.dumps(manifest["config"]))

    async def measured_fixture(config, model_path, directory, metadata):
        engine = config["engine"]["name"]
        shutil.copytree(synthetic_runs / engine, directory / "run")
        path = directory / "run/manifest.json"
        saved = json.loads(path.read_text())
        saved["metadata"].update(metadata)
        path.write_text(json.dumps(saved))
        return {"status": "completed", "error": None}

    monkeypatch.setattr(service, "serve_and_measure", measured_fixture)
    for engine in ("vllm", "sglang"):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "service.py", "--config", str(config_path), "--engine", engine,
                "--model-path", str(tmp_path / "model"), "--output", str(tmp_path / engine),
                "--gpu", "L40S:1", "--cpu-cores", "4", "--memory-mib", "16384",
            ],
        )
        service.main()
    results = compare(tmp_path / "vllm/run", tmp_path / "sglang/run", tmp_path / "comparison")
    assert {r["engine"] for r in results} == {"vllm", "sglang"}
    saved, _ = read_run(tmp_path / "vllm/run")
    assert saved["metadata"]["resources"] == {
        "gpu": "L40S:1", "cpu_cores": 4, "memory_mib": 16384,
    }
