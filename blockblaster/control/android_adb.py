"""Android device control via ADB ``screencap`` (compatibility backend).

This is the legacy / fallback Android backend.  It uses ``adb exec-out
screencap`` which is portable to every Android version but slow on emulators
(~1-2 fps on BlueStacks because each call spawns a new adb subprocess and the
device has to serialize the framebuffer per call).

For higher capture rates use :class:`AndroidScreenrecordDevice`
(:mod:`android_screenrecord`), which streams H.264 instead.
"""

from __future__ import annotations

import struct
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np

from blockblaster.control.adb_utils import (
    ADB_BIN,
    ADB_TIMEOUT,
    adb_run,
    auto_detect_serial,
    parse_wm_size,
    reconnect_tcp,
)
from blockblaster.control.device import Device

_TARGET_FPS  = 10  # background capture rate (faster ≈ more CPU)

# Raw screencap header layout: [width u32][height u32][format u32][reserved u32]
# format 1 = RGBA_8888, 4 = BGRA_8888 — we handle both
_SCREENCAP_HDR = struct.Struct("<IIII")  # 16 bytes


# Re-export ADB_BIN under the old private name for callers that still import it
# (e.g. blockblaster.control.touch_capture).
_ADB_BIN = ADB_BIN


class AndroidAdbDevice(Device):
    """Compatibility Android backend using ``screencap`` (slow but portable).

    Parameters
    ----------
    serial:
        ADB device serial (e.g. ``emulator-5554``).  Auto-detected when ``None``.
    """

    name           = "Android (ADB screencap)"
    supports_input = True

    def __init__(self, serial: Optional[str] = None) -> None:
        self._serial    = serial or auto_detect_serial()
        self._prefix    = [ADB_BIN, "-s", self._serial]
        self._size: Optional[tuple[int, int]] = None

        self._lock      = threading.Lock()
        # Serialises every subprocess call to adb.  BlueStacks (and some
        # emulators) refuse concurrent shell sessions and respond with
        # "error: closed", so the capture thread and the input thread must
        # take turns rather than run in parallel.
        self._adb_lock  = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id  = 0
        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=4.0)
            self._thread = None

    # ── Capture ───────────────────────────────────────────────────────────────

    def get_latest_with_id(self) -> tuple[Optional[np.ndarray], int]:
        with self._lock:
            return self._frame, self._frame_id

    def screen_size(self) -> tuple[int, int]:
        if self._size is None:
            self._size = parse_wm_size(self._serial)
        return self._size

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    # ── Input ─────────────────────────────────────────────────────────────────

    def tap(self, x: int, y: int) -> None:
        adb_run(self._serial, ["shell", "input", "tap", str(x), str(y)])

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        # ``draganddrop`` does a built-in stationary long-press at (x1,y1)
        # before the move, which Block Blast needs to register the piece grab.
        # Plain ``input swipe`` interpolates immediately and never triggers
        # the grab.
        adb_run(self._serial, [
            "shell", "input", "touchscreen", "draganddrop",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms),
        ])

    # ── Internal ──────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        dt = 1.0 / _TARGET_FPS
        while self._running:
            t0 = time.monotonic()
            try:
                frame = self._screencap()
                with self._lock:
                    self._frame    = frame
                    self._frame_id += 1
                    self._last_error = None
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                time.sleep(1.0)
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def _screencap(self) -> np.ndarray:
        """Grab a single frame via ``adb exec-out screencap`` (raw pixels)."""
        data = self._screencap_raw_bytes()
        return _decode_screencap(data)

    def _screencap_raw_bytes(self) -> bytes:
        for attempt in range(2):
            with self._adb_lock:
                result = subprocess.run(
                    self._prefix + ["exec-out", "screencap"],
                    capture_output=True, timeout=ADB_TIMEOUT,
                )
            stderr = result.stderr.decode(errors="replace")
            if result.returncode == 0 and "error: closed" not in stderr:
                return result.stdout
            if attempt == 0 and "error: closed" in stderr:
                print("[adb] screencap connection closed, reconnecting…")
                reconnect_tcp(self._serial)
                time.sleep(0.5)
            else:
                raise RuntimeError(f"screencap failed: {stderr!r}")
        raise RuntimeError("screencap failed after retry")


def _decode_screencap(data: bytes) -> np.ndarray:
    """Decode either raw or PNG screencap output into BGR."""
    if data[:4] == b"\x89PNG":
        buf   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("screencap: failed to decode PNG fallback")
        return frame
    if len(data) < _SCREENCAP_HDR.size:
        raise RuntimeError(f"screencap raw: response too short ({len(data)} bytes)")
    w, h, fmt, _ = _SCREENCAP_HDR.unpack_from(data, 0)
    expected = w * h * 4
    payload  = data[_SCREENCAP_HDR.size:]
    if len(payload) < expected:
        raise RuntimeError(
            f"screencap raw: expected {expected} bytes of pixels, "
            f"got {len(payload)} (w={w} h={h})"
        )
    img = np.frombuffer(payload, dtype=np.uint8, count=expected).reshape((h, w, 4))
    if fmt == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
