"""Abstract Device protocol + factory.

All platform implementations (Android ADB, iOS read-only) satisfy this
interface so the rest of the code can work device-agnostically.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np


class InputNotSupportedError(RuntimeError):
    """Raised when tap/swipe is called on a device that cannot inject input."""


class Device:
    """Base class that all platform back-ends inherit from.

    Sub-classes must implement :meth:`capture`, :meth:`screen_size`,
    :meth:`start`, and :meth:`stop`.  Input methods only need overriding when
    ``supports_input`` is ``True``.
    """

    name: str = "unknown"
    supports_input: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start any background capture threads (idempotent)."""

    def stop(self) -> None:
        """Stop background threads and release resources."""

    # ── Capture ───────────────────────────────────────────────────────────────

    def get_latest_with_id(self) -> tuple[Optional[np.ndarray], int]:
        """Return ``(frame_bgr, frame_id)``.

        ``frame_id`` increments every time a new frame is captured.  Compare
        the id (not the pixel data) to avoid redundant processing.
        Returns ``(None, 0)`` if no frame is available yet.
        """
        raise NotImplementedError

    def screen_size(self) -> tuple[int, int]:
        """Return ``(width_px, height_px)`` of the device screen."""
        raise NotImplementedError

    # ── Input ─────────────────────────────────────────────────────────────────

    @property
    def last_error(self) -> Optional[str]:
        return None

    def tap(self, x: int, y: int) -> None:
        """Tap the screen at ``(x, y)`` in device pixels."""
        if not self.supports_input:
            raise InputNotSupportedError(
                f"{self.name} does not support touch injection. "
                "Use an Android device (ADB) for auto-play."
            )

    def swipe(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 300,
    ) -> None:
        """Drag from ``(x1, y1)`` to ``(x2, y2)`` over ``duration_ms``."""
        if not self.supports_input:
            raise InputNotSupportedError(
                f"{self.name} does not support touch injection. "
                "Use an Android device (ADB) for auto-play."
            )


def device_status_detail(
    device: Device, frame_w: int = 0, frame_h: int = 0,
) -> str:
    """Short summary for the status bar / connect log (backend, serial, size)."""
    parts = [device.name]
    serial = getattr(device, "_serial", None)
    if serial:
        parts.append(str(serial))
    if frame_w > 0 and frame_h > 0:
        parts.append(f"{frame_w}×{frame_h}")
    parts.append("input" if device.supports_input else "read-only")
    return " · ".join(parts)


def make_device(
    platform: Literal["ios", "android"],
    serial: Optional[str] = None,
) -> Device:
    """Instantiate the correct :class:`Device` for *platform*.

    Parameters
    ----------
    platform:
        ``"ios"``     – read-only iOS mirror via tunneld / DVT.
        ``"android"`` – full control via ADB (emulator or USB phone).
    serial:
        Android only.  ADB device serial; auto-detected if ``None``.
    """
    if platform == "ios":
        from blockblaster.control.ios_readonly import IosReadOnlyDevice
        return IosReadOnlyDevice()
    if platform == "android":
        # TEMP: forcing slow screencap backend to test whether screenrecord
        # interferes with input injection.  Flip USE_SCREENRECORD back to True
        # to re-enable the fast H.264 path.
        USE_SCREENRECORD = True
        if USE_SCREENRECORD:
            try:
                from blockblaster.control.android_screenrecord import (
                    AndroidScreenrecordDevice,
                )
                return AndroidScreenrecordDevice(serial=serial)
            except Exception as exc:
                print(
                    f"[device] screenrecord backend unavailable ({exc}); "
                    "falling back to slow screencap backend."
                )
        from blockblaster.control.android_adb import AndroidAdbDevice
        return AndroidAdbDevice(serial=serial)
    raise ValueError(f"Unknown platform {platform!r}. Choose 'ios' or 'android'.")
