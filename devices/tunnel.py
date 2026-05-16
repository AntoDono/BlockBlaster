"""Run ``pymobiledevice3 remote tunneld`` under sudo and clean up on exit.

This script is just a convenience wrapper so you don't have to remember the
exact tunneld invocation (and the sudo prompt). Any DVT consumer you run
alongside it (``blockblaster/assist/app.py``, ``devices/test.py``, ...) will
keep the QUIC tunnel alive on its own by polling DVT continuously.

Earlier versions of this script also held a DVT screenshot session open as a
keepalive. That conflicts with ``app.py`` (iOS only tolerates one DVT consumer
per device at a time), and shows up as endless "Created tunnel … Disconnected
from tunnel" pairs in tunneld's logs. Don't reintroduce it.

Usage:
    sudo uv run ./devices/tunnel.py
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys

TUNNELD_READY_LINE = "Application startup complete"
TUNNELD_READY_TIMEOUT_S = 30.0

TUNNELD_CMD = [
    "sudo",
    "uv",
    "run",
    "python",
    "-m",
    "pymobiledevice3",
    "remote",
    "tunneld",
]


async def _start_tunneld() -> asyncio.subprocess.Process:
    print(f"tunnel: launching {' '.join(TUNNELD_CMD)}", flush=True)
    proc = await asyncio.create_subprocess_exec(
        *TUNNELD_CMD,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # own process group so we can kill the tree
    )

    ready = asyncio.Event()

    async def _watch() -> None:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                return
            text = line.decode(errors="replace")
            sys.stdout.write(f"[tunneld] {text}")
            sys.stdout.flush()
            if TUNNELD_READY_LINE in text:
                ready.set()

    asyncio.create_task(_watch())

    try:
        await asyncio.wait_for(ready.wait(), timeout=TUNNELD_READY_TIMEOUT_S)
    except asyncio.TimeoutError:
        await _terminate(proc)
        raise RuntimeError("tunneld did not become ready in time")

    print("tunnel: tunneld is up (Ctrl+C to stop)", flush=True)
    return proc


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        await proc.wait()


async def main() -> None:
    proc = await _start_tunneld()
    try:
        await proc.wait()
        print(
            f"tunnel: tunneld exited (code {proc.returncode})",
            flush=True,
        )
    finally:
        await _terminate(proc)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
