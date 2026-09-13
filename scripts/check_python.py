"""Check local dependency files before starting imports that could wait on cloud sync."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Darwin's SF_DATALESS flag is absent from stat on some Python builds.
SF_DATALESS = getattr(stat, "SF_DATALESS", 0x40000000)


def offloaded_files(environment: Path) -> list[Path]:
    """Inspect metadata only; do not read offloaded file contents."""
    result = []
    for directory, _, names in os.walk(environment):
        for name in names:
            path = Path(directory) / name
            info = path.lstat()
            if info.st_size and getattr(info, "st_flags", 0) & SF_DATALESS:
                result.append(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=REPOSITORY_ROOT / ".venv")
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    environment = args.environment.resolve()
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    print(f"Environment: {environment}", flush=True)
    if not python.exists():
        print("Missing Python. Install the locked environment before running checks.")
        return 1
    free = shutil.disk_usage(environment).free / 1024**3
    print(f"Available disk space: {free:.1f} GiB", flush=True)
    offloaded = offloaded_files(environment)
    if offloaded:
        print(f"Found {len(offloaded)} offloaded files; imports may wait for downloads.")
        for path in offloaded[:5]:
            print(f"  {path.relative_to(environment)}")
        print("See DEVELOPMENT.md: Keeping Python dependencies local on macOS.")
        return 1
    print("No offloaded dependency files found. Checking imports...", flush=True)
    code = (
        "import time; start = time.monotonic(); "
        "import pytest, torch, transformers; "
        "print(f'Imports completed in {time.monotonic() - start:.2f}s'); "
        "print('pytest', pytest.__version__, 'torch', torch.__version__, "
        "'transformers', transformers.__version__)"
    )
    try:
        result = subprocess.run([str(python), "-I", "-c", code], timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print(f"Imports exceeded {args.timeout:g}s. No files or settings were changed.")
        return 1
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
