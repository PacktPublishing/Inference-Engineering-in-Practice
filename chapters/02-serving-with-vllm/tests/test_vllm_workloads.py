import json

import pytest
from vllm_workloads import (
    Workload,
    WorkloadRequest,
    iter_request_ids,
    make_concurrency_workload,
    make_prefix_workload,
    workload_document,
    write_workload_jsonl,
)


def test_concurrency_workload_is_deterministic_and_mixed_length() -> None:
    first = make_concurrency_workload(
        count=4,
        max_concurrency=2,
        output_tokens=(8, 32),
    )
    second = make_concurrency_workload(
        count=4,
        max_concurrency=2,
        output_tokens=(8, 32),
    )

    assert first == second
    assert [request.max_tokens for request in first.requests] == [8, 32, 8, 32]
    assert first.max_concurrency == 2
    assert workload_document(first)["schema_version"] == 1


def test_prefix_workload_reuses_the_exact_system_prompt() -> None:
    workload = make_prefix_workload(
        shared_system_prompt="Shared instructions.",
        questions=("Question one?", "Question two?"),
    )

    assert {request.system_prompt for request in workload.requests} == {
        "Shared instructions."
    }
    assert workload.requests[0].messages() == [
        {"role": "system", "content": "Shared instructions."},
        {"role": "user", "content": "Question one?"},
    ]


def test_request_payload_omits_stream_options_when_not_streaming() -> None:
    request = WorkloadRequest(
        request_id="one",
        user_prompt="Hello",
        max_tokens=4,
    )

    payload = request.api_payload(model="served-name", stream=False)

    assert payload["stream"] is False
    assert "stream_options" not in payload


def test_open_loop_policy_and_global_ids_are_validated() -> None:
    request = WorkloadRequest("duplicate", "Hello", 4)
    with pytest.raises(ValueError, match="request_rate"):
        Workload(
            name="open-loop",
            requests=(request,),
            arrival_mode="open_loop",
        )
    with pytest.raises(ValueError, match="arrival_mode"):
        Workload(
            name="invalid",
            requests=(request,),
            arrival_mode="burst",
        )

    first = Workload(name="first", requests=(request,))
    second = Workload(name="second", requests=(request,))
    with pytest.raises(ValueError, match="unique across"):
        iter_request_ids([first, second])


def test_jsonl_contains_self_describing_api_requests(tmp_path) -> None:
    workload = make_concurrency_workload(
        count=2,
        max_concurrency=1,
        output_tokens=(8,),
    )
    destination = tmp_path / "requests.jsonl"

    write_workload_jsonl(workload, destination, model="served-name")

    rows = [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2
    assert rows[0]["workload"] == "concurrency-sweep"
    assert rows[0]["payload"]["model"] == "served-name"
    assert rows[0]["payload"]["stream_options"] == {"include_usage": True}

    invalid_destination = tmp_path / "invalid.jsonl"
    with pytest.raises(ValueError, match="model"):
        write_workload_jsonl(workload, invalid_destination, model="")
    assert not invalid_destination.exists()
