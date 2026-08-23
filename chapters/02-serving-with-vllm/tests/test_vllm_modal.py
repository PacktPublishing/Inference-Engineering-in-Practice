import pytest
from model_config import MODEL_ID as CHAPTER_1_MODEL_ID
from model_config import MODEL_REVISION as CHAPTER_1_MODEL_REVISION
from vllm_config import (
    DEFAULT_SERVER_CONFIG,
    HF_CACHE_PATH,
    MODAL_MAX_CONTAINERS,
    MODAL_MIN_CONTAINERS,
    MODAL_SCALEDOWN_WINDOW_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    SERVED_MODEL_NAME,
    VLLM_CACHE_PATH,
    ServerConfig,
)
from vllm_modal import build_serve_command, server_environment


def _flag_value(command: tuple[str, ...], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def test_default_command_is_pinned_and_bounded() -> None:
    command = build_serve_command()

    assert command[:3] == ("vllm", "serve", MODEL_ID)
    assert _flag_value(command, "--revision") == MODEL_REVISION
    assert _flag_value(command, "--served-model-name") == SERVED_MODEL_NAME
    assert _flag_value(command, "--max-model-len") == str(
        DEFAULT_SERVER_CONFIG.max_model_len
    )
    assert "--enable-per-request-metrics" in command
    assert "--enable-mfu-metrics" in command
    assert "--enable-prefix-caching" not in command
    assert MODAL_MIN_CONTAINERS == 0
    assert MODAL_MAX_CONTAINERS == 1
    assert MODAL_SCALEDOWN_WINDOW_SECONDS == 60


def test_baseline_model_matches_chapter_one() -> None:
    assert MODEL_ID == CHAPTER_1_MODEL_ID
    assert MODEL_REVISION == CHAPTER_1_MODEL_REVISION


def test_optional_flags_and_extra_arguments_are_explicit() -> None:
    config = ServerConfig(
        enable_prefix_caching=True,
        enable_per_request_metrics=False,
        enable_mfu_metrics=False,
        extra_args=("--max-num-batched-tokens", "2048"),
    )

    command = build_serve_command(config)

    assert "--enable-prefix-caching" in command
    assert "--enable-per-request-metrics" not in command
    assert "--enable-mfu-metrics" not in command
    assert command[-2:] == ("--max-num-batched-tokens", "2048")


def test_server_environment_uses_separate_caches() -> None:
    environment = server_environment({"EXPERIMENT_ID": "test-run"})

    assert environment["HF_HOME"] == HF_CACHE_PATH
    assert environment["VLLM_CACHE_ROOT"] == VLLM_CACHE_PATH
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert environment["EXPERIMENT_ID"] == "test-run"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("port", 0, "port"),
        ("max_model_len", 0, "max_model_len"),
        ("max_num_seqs", 0, "max_num_seqs"),
        ("gpu_memory_utilization", 1.1, "gpu_memory_utilization"),
    ],
)
def test_invalid_server_configuration_is_rejected(
    field: str,
    value: int | float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ServerConfig(**{field: value})
