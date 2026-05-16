"""Background thread that continuously grabs frames from an iOS device via tunneld/DVT."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Allow importing from the devices/ sibling folder at the project root.
_DEVICES_DIR = Path(__file__).resolve().parents[2] / "devices"
if str(_DEVICES_DIR) not in sys.path:
    sys.path.insert(0, str(_DEVICES_DIR))

from get_frame import get_frame, open_device_stream  # type: ignore[import]

TARGET_FPS = 15
_FRAME_DT = 1.0 / TARGET_FPS


class DeviceStream:
    """Runs an asyncio capture loop in a daemon thread.

    Call start() once, poll get_latest() from the pygame thread, call stop()
    before exit.  All errors are caught and surfaced via last_error.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background capture thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self.last_error = None
        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to exit and wait for it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_latest(self) -> Optional[np.ndarray]:
        """Return the most recent BGR frame (or None if not yet available)."""
        with self._lock:
            return self._frame

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._capture_loop())
        except Exception as exc:
            with self._lock:
                self.last_error = str(exc)

    async def _capture_loop(self) -> None:
        try:
            async with open_device_stream() as screenshot:
                self.last_error = None
                next_deadline = asyncio.get_event_loop().time()
                while self._running:
                    frame = await get_frame(screenshot)
                    if frame is not None:
                        with self._lock:
                            self._frame = frame

                    next_deadline += _FRAME_DT
                    sleep_for = next_deadline - asyncio.get_event_loop().time()
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                    else:
                        next_deadline = asyncio.get_event_loop().time()
        except Exception as exc:
            with self._lock:
                self.last_error = str(exc)
            # brief pause so the thread doesn't spin on repeated errors
            time.sleep(1.0)
