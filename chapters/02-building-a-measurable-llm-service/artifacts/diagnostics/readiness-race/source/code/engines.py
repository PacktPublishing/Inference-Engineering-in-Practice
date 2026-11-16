"""The two concrete engine launch commands used in this chapter."""

from pathlib import Path

IMAGES = {
    "vllm": "vllm/vllm-openai@"
    "sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b",
    "sglang": "lmsysorg/sglang@"
    "sha256:37bbbd3444732a464bbc68dee4fb0164e0ce9e18e2f027f3fc967f1152d3c262",
}
VERSIONS = {"vllm": "0.29.0", "sglang": "0.5.19"}


def engine_name(config):
    name = config.get("engine", {}).get("name", "vllm")
    if name not in VERSIONS:
        raise ValueError("This chapter implements vllm and sglang")
    return name


def server_command(config: dict, model_path: Path) -> list[str]:
    name = engine_name(config)
    if name == "vllm":
        command = [
            "vllm",
            "serve",
            str(model_path),
            "--served-model-name",
            config["model"],
            "--tokenizer",
            str(model_path),
            "--dtype",
            "bfloat16",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--max-model-len",
            "1024",
            "--max-num-seqs",
            "64",
            "--gpu-memory-utilization",
            "0.75",
            "--no-enable-prefix-caching",
            "--generation-config",
            "vllm",
            "--seed",
            "17",
        ]
        if config.get("engine", {}).get("enforce_eager", False):
            command.append("--enforce-eager")
        return command
    if config.get("engine", {}).get("enforce_eager", False):
        raise ValueError("The SGLang example uses standard execution")
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--tokenizer-path",
        str(model_path),
        "--served-model-name",
        config["model"],
        "--dtype",
        "bfloat16",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--context-length",
        "1024",
        "--max-running-requests",
        "64",
        "--mem-fraction-static",
        "0.75",
        "--disable-radix-cache",
        "--sampling-defaults",
        "openai",
        "--random-seed",
        "17",
        "--enable-metrics",
    ]
