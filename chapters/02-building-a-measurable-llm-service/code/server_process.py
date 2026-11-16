"""Bound a local API-server process, including failed startup and teardown."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
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
    readiness_url: str | None = None,
    environment: dict[str, str] | None = None,
):
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, **(environment or {})},
        )
        try:
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                try:
                    async with asyncio.timeout(readiness_s):
                        while True:
                            if process.poll() is not None:
                                raise RuntimeError(
                                    f"Server exited during startup: {process.returncode}"
                                )
                            try:
                                response = await client.get(
                                    readiness_url or base_url.rstrip("/") + "/models"
                                )
                                if response.status_code == 200 and (
                                    readiness_url is not None or response.json().get("data")
                                ):
                                    break
                            except (httpx.HTTPError, ValueError):
                                pass
                            await asyncio.sleep(0.25)
                except TimeoutError as exc:
                    raise TimeoutError("Server readiness deadline expired") from exc
            yield process
        finally:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                await asyncio.to_thread(process.wait, timeout=shutdown_s)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # Also run if cancellation interrupts the graceful wait. The parent
                # can exit before its workers, so always clean the whole group.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                # No await here: another cancellation must not interrupt reaping.
                # SIGKILL has already been sent; bound this final synchronous wait.
                process.wait(timeout=shutdown_s)
