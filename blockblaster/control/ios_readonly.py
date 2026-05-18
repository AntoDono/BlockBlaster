"""iOS read-only device: wraps the existing tunneld/DVT DeviceStream."""

from __future__ import annotations

from typing import Optional

import numpy as np

from blockblaster.assist.device_stream import DeviceStream
from blockblaster.control.device import Device, InputNotSupportedError


class IosReadOnlyDevice(Device):
    """Live iOS screen mirror with no touch-injection capability.

    Wraps :class:`~blockblaster.assist.device_stream.DeviceStream` to satisfy
    the :class:`~blockblaster.control.device.Device` interface.
    """

    name           = "iOS (read-only)"
    supports_input = False

    def __init__(self) -> None:
        self._stream   = DeviceStream()
        self._size: Optional[tuple[int, int]] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()

    # ── Capture ───────────────────────────────────────────────────────────────

    def get_latest_with_id(self) -> tuple[Optional[np.ndarray], int]:
        frame, fid = self._stream.get_latest_with_id()
        if frame is not None and self._size is None:
            h, w    = frame.shape[:2]
            self._size = (w, h)
        return frame, fid

    def screen_size(self) -> tuple[int, int]:
        if self._size is not None:
            return self._size
        frame, _ = self._stream.get_latest_with_id()
        if frame is None:
            return (0, 0)
        h, w = frame.shape[:2]
        self._size = (w, h)
        return self._size

    @property
    def last_error(self) -> Optional[str]:
        return self._stream.last_error

    # ── Input ─────────────────────────────────────────────────────────────────

    def tap(self, x: int, y: int) -> None:
        raise InputNotSupportedError(
            "iOS touch injection is not supported.\n"
            "Use --platform android for auto-play."
        )

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        raise InputNotSupportedError(
            "iOS touch injection is not supported.\n"
            "Use --platform android for auto-play."
        )
