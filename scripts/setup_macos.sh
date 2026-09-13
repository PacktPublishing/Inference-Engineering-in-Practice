#!/bin/sh
# Keep this project's disposable dependencies outside cloud-managed Documents.
set -eu
if [ "$(uname -s)" != Darwin ]; then
    echo 'This helper is for macOS. Elsewhere, use uv sync --locked.' >&2
    exit 1
fi
project_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
environment_root="${XDG_DATA_HOME:-$HOME/.local/share}/venvs/llm-inference-engineering"
cd "$project_root"
if [ -e .venv ] || [ -L .venv ]; then
    if [ ! -L .venv ] || [ "$(readlink .venv)" != "$environment_root" ]; then
        echo 'An existing .venv uses a different location.' >&2
        echo 'Preserve or remove that disposable environment before running this helper.' >&2
        echo 'This helper has not changed it.' >&2
        exit 1
    fi
fi
UV_PROJECT_ENVIRONMENT="$environment_root" uv sync --locked --python 3.12
if [ ! -L .venv ]; then
    ln -s "$environment_root" .venv
fi
.venv/bin/python scripts/check_python.py
