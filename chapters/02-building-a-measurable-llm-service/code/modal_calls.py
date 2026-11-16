"""Bound the adapter's wait, including scheduling; preserve uncertain cleanup."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path


async def invoke(function, *args, wait_s: float, record: Path, cancel_s: float = 30):
    """Submit once. On failure, request termination without claiming verified cleanup.

    A lost submission response can leave a remote call without a local handle.
    In that case, stop the ephemeral app and inspect its tasks explicitly.
    These local records describe provisioning, never measured engine outcomes.
    """
    state = {
        "started_utc": datetime.now(UTC).isoformat(),
        "wait_s": wait_s,
        "cancel_s": cancel_s,
        "status": "submitting",
        "call_id": None,
        "cleanup_verified": False,
    }

    def save():
        record.write_text(json.dumps(state, indent=2) + "\n")

    save()
    call = None
    try:
        async with asyncio.timeout(wait_s):
            call = await function.spawn.aio(*args)
            state.update(status="waiting", call_id=call.object_id)
            save()
            result = await call.get.aio(timeout=wait_s)
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        # Cancellation must still run if saving the failure record fails.
        try:
            save()
        finally:
            if call is not None:
                try:
                    async with asyncio.timeout(cancel_s):
                        await call.cancel.aio(terminate_containers=True)
                    state["cancellation"] = "acknowledged; verify app tasks separately"
                except BaseException as cancel_error:
                    state["cancellation"] = f"unverified: {type(cancel_error).__name__}"
            else:
                state["cancellation"] = "no call handle; stop and inspect the app"
            state["ended_utc"] = datetime.now(UTC).isoformat()
            save()
        raise
    state.update(status="returned", ended_utc=datetime.now(UTC).isoformat())
    save()
    return result
