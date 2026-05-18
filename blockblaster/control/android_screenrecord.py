"""Fast Android capture using ``adb shell screenrecord`` + PyAV.

``screencap`` (the slow backend in :mod:`android_adb`) takes ~700 ms/frame on
BlueStacks because it does a full framebuffer readback + serialization per
request.  ``screenrecord`` instead pipes a continuous H.264 stream from the
device, which we decode locally with PyAV.  Expected throughput: 30-60 fps.

``screenrecord`` has two quirks we work around:
1. Hard 180-second segment limit per invocation — we restart in a loop.
2. No I-frame on start — first decoded frame can take ~0.5 s.  Subsequent
   frames are near-zero-latency.

Input (tap / swipe) reuses the existing ADB shell commands.
"""

from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager
from typing import Optional

import av
import cv2
import numpy as np

from blockblaster.control.adb_utils import (
    ADB_BIN,
    adb_run,
    auto_detect_serial,
    parse_wm_size,
    reconnect_tcp,
)
from blockblaster.control.device import Device

_SEGMENT_SECONDS = 170          # under the 180-second screenrecord hard cap
_BIT_RATE        = 8_000_000    # 8 Mbps — plenty for an 8x8 grid game
_DIAG_INTERVAL_S = 2.0          # how often to log capture FPS


class AndroidScreenrecordDevice(Device):
    """Fast Android backend: H.264 stream from ``screenrecord`` decoded via PyAV.

    Parameters
    ----------
    serial:
        ADB device serial.  Auto-detected if ``None``.
    max_size:
        Optional cap on the longer screen dimension (e.g. 1280) passed to
        ``screenrecord --size``.  Smaller = faster encode on the device.
        ``None`` means use the device's native resolution.
    """

    name           = "Android (screenrecord)"
    supports_input = True

    def __init__(
        self,
        serial: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        self._serial   = serial or auto_detect_serial()
        self._max_size = max_size
        self._size: Optional[tuple[int, int]] = None
        self._record_size: Optional[tuple[int, int]] = None

        self._lock     = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._proc:    Optional[subprocess.Popen] = None
        self._last_error: Optional[str] = None

        # Pause gate so the capture loop blocks while a swipe is in flight.
        # Reason: in the old AndroidAdbDevice, ``screencap`` and ``input
        # swipe`` were serialized by ``_adb_lock`` so the device only ever
        # saw one shell session at a time.  With the persistent
        # ``screenrecord`` subprocess, ``input swipe`` runs concurrently and
        # One UI's GestureDetector reclassifies the contended swipe as a
        # system gesture.  Killing the screenrecord segment for the
        # duration of the swipe restores the old serialized behavior.
        self._allow_capture = threading.Event()
        self._allow_capture.set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, name="screenrecord", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None
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
        # See AndroidAdbDevice.swipe for rationale.
        adb_run(self._serial, [
            "shell", "input", "touchscreen", "draganddrop",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms),
        ])

    @contextmanager
    def _paused_capture(self):  # type: ignore[no-untyped-def]
        """Kill the current screenrecord segment, run the body, then resume."""
        self._allow_capture.clear()
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
        try:
            yield
        finally:
            time.sleep(0.05)
            self._allow_capture.set()


    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_record_size(self) -> tuple[int, int]:
        """Pick the ``--size WxH`` argument for screenrecord."""
        if self._record_size is not None:
            return self._record_size
        w, h = self.screen_size()
        if self._max_size is not None and max(w, h) > self._max_size:
            scale = self._max_size / max(w, h)
            w = int(round(w * scale)) & ~1   # keep even
            h = int(round(h * scale)) & ~1
        self._record_size = (w, h)
        return self._record_size

    def _spawn_screenrecord(self) -> subprocess.Popen:
        w, h = self._resolve_record_size()
        cmd = [
            ADB_BIN, "-s", self._serial, "exec-out",
            "screenrecord",
            "--output-format=h264",
            f"--size={w}x{h}",
            f"--bit-rate={_BIT_RATE}",
            f"--time-limit={_SEGMENT_SECONDS}",
            "-",
        ]
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
        )

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        while self._running:
            # Block while a swipe is in flight so we don't re-spawn
            # screenrecord on top of the in-progress input swipe.
            if not self._allow_capture.wait(timeout=0.5):
                continue
            try:
                self._run_one_segment()
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                with self._lock:
                    self._last_error = str(exc)
                # Segment errors right after a swipe-induced kill are
                # expected — only log on repeated failures.
                if consecutive_failures >= 2:
                    print(f"[screenrecord] segment error: {exc}")
                if consecutive_failures >= 3:
                    reconnect_tcp(self._serial)
                    time.sleep(1.0)
                else:
                    time.sleep(0.1)

    def _run_one_segment(self) -> None:
        """Stream one ``screenrecord`` invocation through PyAV until it ends."""
        self._proc = self._spawn_screenrecord()
        assert self._proc.stdout is not None

        diag_n      = 0
        diag_window = time.monotonic()

        # PyAV reads raw H.264 NAL units directly from the pipe.
        try:
            container = av.open(self._proc.stdout, format="h264", mode="r")
        except av.error.OSError as exc:
            raise RuntimeError(f"PyAV could not open screenrecord stream: {exc}")

        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"   # multi-threaded decode

            for frame in container.decode(stream):
                if not self._running:
                    break
                arr = frame.to_ndarray(format="bgr24")
                with self._lock:
                    self._frame    = arr
                    self._frame_id += 1
                    self._last_error = None
                diag_n += 1

                now = time.monotonic()
                if now - diag_window >= _DIAG_INTERVAL_S:
                    fps = diag_n / (now - diag_window)
                    print(f"[screenrecord] {diag_n} frames in "
                          f"{now - diag_window:.2f}s → {fps:.1f} fps")
                    diag_n      = 0
                    diag_window = now
        finally:
            try:
                container.close()
            except Exception:
                pass
            if self._proc is not None:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                self._proc.wait(timeout=2.0)
                self._proc = None
