"""Bound a local API-server process, including failed startup and teardown."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import httpx


@asynccontextmanager
async def running_server(
    command: list[str],
    base_url: str,
    log_path: Path,
    readiness_s: float = 300,
    shutdown_s: float = 10,
):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                deadline = time.perf_counter() + readiness_s
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited during startup: {process.returncode}")
                    if time.perf_counter() >= deadline:
                        raise TimeoutError("Server readiness deadline expired")
                    try:
                        response = await client.get(base_url.rstrip("/") + "/models")
                        if response.status_code == 200 and response.json().get("data"):
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    await asyncio.sleep(min(0.25, max(0, deadline - time.perf_counter())))
            yield process
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                await asyncio.to_thread(process.wait, timeout=shutdown_s)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                await asyncio.to_thread(process.wait)
            # The parent can exit before its engine workers; clean the whole group.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
