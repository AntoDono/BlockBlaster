"""ADB getevent-based touch-event capture for human-guided calibration.

``TouchCapture`` spawns ``adb shell getevent`` on a background daemon thread,
parses ABS_MT_POSITION_X/Y and BTN_TOUCH events, and exposes each completed
human swipe as a ``(down_px, up_px, t_down, t_up)`` named-tuple with
coordinates already scaled to screen pixels.

Usage::

    cap = TouchCapture(serial="emulator-5554", screen_w=1080, screen_h=1920)
    cap.start()
    swipe = cap.wait_for_swipe(timeout_s=30, should_continue=lambda: True)
    if swipe:
        print(swipe.up_px)   # screen-space (x, y) where finger lifted
    cap.stop()

Coordinate discovery
--------------------
``_discover_touch_device`` runs ``adb shell getevent -lp``, finds the first
input device that reports ``ABS_MT_POSITION_X`` **and** ``ABS_MT_POSITION_Y``,
and extracts the ``min`` / ``max`` raw values for both axes.  These are used
to scale raw hardware units to screen pixels.

If no multi-touch device is found the function falls back to the first device
reporting ``ABS_X`` + ``ABS_Y`` (single-touch protocol A).

BlueStacks / emulator compatibility
------------------------------------
BlueStacks 5 exposes a virtual multi-touch digitiser.  ``getevent -lp``
output on real devices uses labels like ``ABS_MT_POSITION_X``; some emulators
print different capitalisations or abbreviations.  All label matching is
done case-insensitively.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

from blockblaster.control.adb_utils import ADB_BIN as _ADB_BIN


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class _AxisRange:
    minimum: int
    maximum: int

    def scale(self, raw: int, screen_dim: int) -> int:
        span = max(1, self.maximum - self.minimum)
        return int(round((raw - self.minimum) * screen_dim / span))


class Swipe(NamedTuple):
    down_px: tuple[int, int]
    up_px:   tuple[int, int]
    t_down:  float   # monotonic seconds
    t_up:    float


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------

_MT_X_RE  = re.compile(r"ABS_MT_POSITION_X", re.IGNORECASE)
_MT_Y_RE  = re.compile(r"ABS_MT_POSITION_Y", re.IGNORECASE)
_ABS_X_RE = re.compile(r"\bABS_X\b",         re.IGNORECASE)
_ABS_Y_RE = re.compile(r"\bABS_Y\b",         re.IGNORECASE)
_RANGE_RE = re.compile(r"min\s+(\d+),\s+max\s+(\d+)", re.IGNORECASE)
_DEV_RE   = re.compile(r"^add\s+device\s+\d+:\s*(/dev/input/\S+)", re.MULTILINE)


def _discover_touch_device(
    serial: str,
) -> tuple[str, _AxisRange, _AxisRange]:
    """Return (device_path, x_range, y_range) for the best touch input device.

    Raises ``RuntimeError`` if no suitable device is found.
    """
    raw = subprocess.run(
        [_ADB_BIN, "-s", serial, "shell", "getevent", "-lp"],
        capture_output=True, text=True, timeout=10,
    ).stdout

    # Split into per-device blocks
    blocks: list[str] = []
    current_start = 0
    for m in _DEV_RE.finditer(raw):
        if blocks or m.start() > 0:
            blocks.append(raw[current_start : m.start()])
        current_start = m.start()
    blocks.append(raw[current_start:])

    def _extract_range(block: str, pattern: re.Pattern) -> Optional[_AxisRange]:
        for line in block.splitlines():
            if pattern.search(line):
                rm = _RANGE_RE.search(line)
                if rm:
                    return _AxisRange(int(rm.group(1)), int(rm.group(2)))
        return None

    # Prefer multi-touch (Protocol B)
    for block in blocks:
        dm = _DEV_RE.search(block)
        if not dm:
            continue
        dev = dm.group(1)
        if _MT_X_RE.search(block) and _MT_Y_RE.search(block):
            xr = _extract_range(block, _MT_X_RE)
            yr = _extract_range(block, _MT_Y_RE)
            if xr and yr:
                print(f"[touch_capture] using MT device {dev}  "
                      f"x=[{xr.minimum},{xr.maximum}] y=[{yr.minimum},{yr.maximum}]")
                return dev, xr, yr

    # Fall back to single-touch (Protocol A)
    for block in blocks:
        dm = _DEV_RE.search(block)
        if not dm:
            continue
        dev = dm.group(1)
        if _ABS_X_RE.search(block) and _ABS_Y_RE.search(block):
            xr = _extract_range(block, _ABS_X_RE)
            yr = _extract_range(block, _ABS_Y_RE)
            if xr and yr:
                print(f"[touch_capture] using single-touch device {dev}")
                return dev, xr, yr

    raise RuntimeError(
        "No suitable touch input device found via 'adb shell getevent -lp'.\n"
        "Ensure the Android device is connected and USB debugging is enabled."
    )


# ---------------------------------------------------------------------------
# TouchCapture
# ---------------------------------------------------------------------------

# getevent -lt output has two common formats depending on Android version / emulator:
#
# With '/' separator (AOSP real devices):
#   [  1234.567890] /dev/input/event5: EV_ABS / ABS_MT_POSITION_X   000004a3
#   [  1234.567892] /dev/input/event5: EV_KEY / BTN_TOUCH            DOWN
#
# Without '/' separator (BlueStacks, some emulators):
#   [  1234.567890] /dev/input/event5: EV_ABS       ABS_MT_POSITION_X   000004a3
#   [  1234.567892] /dev/input/event5: EV_KEY       BTN_TOUCH            DOWN
#
# Timestamps are optional (if -t flag unsupported, no [ ... ] prefix).
# The regex below handles all four combinations.
_EV_LINE_RE = re.compile(
    r"(?:\[[\s\d.]+\]\s+)?"       # optional [ timestamp ]
    r"(?:\S+:\s+)?"               # optional device path + colon (omitted when getevent
                                  #   is called with an explicit device path argument)
    r"(\S+)\s+"                   # ev_type  (group 1)
    r"(?:/\s+)?"                  # optional '/ ' separator (present on real devices,
                                  #   absent on BlueStacks / some emulators)
    r"(\S+)\s+"                   # ev_code  (group 2)
    r"(\S+)",                     # ev_value (group 3)
    re.IGNORECASE,
)

# Debug: print the first N raw lines so format issues are immediately visible
_DEBUG_RAW_LINES = 60


class TouchCapture:
    """Capture human finger swipes via ``adb shell getevent``.

    Call :meth:`start` to begin streaming, :meth:`wait_for_swipe` to block
    for one completed drag, and :meth:`stop` to clean up.
    """

    def __init__(self, serial: str, screen_w: int, screen_h: int) -> None:
        self._serial   = serial
        self._sw       = screen_w
        self._sh       = screen_h

        dev, xr, yr    = _discover_touch_device(serial)
        self._dev      = dev
        self._xr       = xr
        self._yr       = yr

        self._lock     = threading.Lock()
        self._swipes: deque[Swipe] = deque()
        self._event    = threading.Event()

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running  = False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._proc = subprocess.Popen(
            [_ADB_BIN, "-s", self._serial, "shell", "getevent", "-lt", self._dev],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr so startup errors are visible
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(
            target=self._reader_loop, name="touch-capture", daemon=True,
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
            self._thread.join(timeout=3.0)
            self._thread = None
        self._event.set()   # unblock any waiting caller

    def wait_for_swipe(
        self,
        timeout_s: float = 30.0,
        should_continue: Callable[[], bool] = lambda: True,
    ) -> Optional[Swipe]:
        """Block until a new swipe is captured, then return it.

        Polls ``should_continue()`` and checks for new swipes every 0.1 s.
        Returns ``None`` if the timeout expires or ``should_continue()``
        becomes False.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and should_continue():
            self._event.wait(timeout=0.1)
            self._event.clear()
            with self._lock:
                if self._swipes:
                    return self._swipes.popleft()
        return None

    # ── Internal ────────────────────────────────────────────────────────

    def _reader_loop(self) -> None:
        """Read getevent stdout, parse events, emit Swipe objects."""
        cur_x:  Optional[int]   = None
        cur_y:  Optional[int]   = None
        down_x: Optional[int]   = None
        down_y: Optional[int]   = None
        t_down: Optional[float] = None
        touch_down: bool        = False
        raw_count: int          = 0   # counter for initial debug dump

        assert self._proc is not None
        assert self._proc.stdout is not None

        try:
            for line in self._proc.stdout:
                if not self._running:
                    break

                # Print the first N raw lines so format issues are visible
                if raw_count < _DEBUG_RAW_LINES:
                    print(f"[touch_capture] raw: {line.rstrip()}")
                    raw_count += 1

                m = _EV_LINE_RE.search(line)
                if not m:
                    continue
                ev_type = m.group(1).upper()
                ev_code = m.group(2).upper()
                ev_val  = m.group(3).upper()

                if ev_type == "EV_ABS":
                    try:
                        raw = int(ev_val, 16)
                    except ValueError:
                        continue

                    if "ABS_MT_POSITION_X" in ev_code or ev_code == "ABS_X":
                        cur_x = self._xr.scale(raw, self._sw)
                        # Latch first position after touch-down
                        if touch_down and down_x is None:
                            down_x = cur_x

                    elif "ABS_MT_POSITION_Y" in ev_code or ev_code == "ABS_Y":
                        cur_y = self._yr.scale(raw, self._sh)
                        if touch_down and down_y is None:
                            down_y = cur_y

                    elif "ABS_MT_TRACKING_ID" in ev_code:
                        # Protocol B: 0xFFFFFFFF == finger lifted
                        if raw == 0xFFFFFFFF or raw == 0xFFFF:
                            if touch_down and cur_x is not None and cur_y is not None:
                                touch_down = False
                                swipe = Swipe(
                                    down_px=(down_x or cur_x, down_y or cur_y),
                                    up_px  =(cur_x, cur_y),
                                    t_down =t_down or time.monotonic(),
                                    t_up   =time.monotonic(),
                                )
                                with self._lock:
                                    self._swipes.append(swipe)
                                self._event.set()
                                print(
                                    f"[touch_capture] swipe via TRACKING_ID: "
                                    f"down={swipe.down_px} up={swipe.up_px}"
                                )
                                down_x = down_y = t_down = None
                        else:
                            # New finger contact
                            if not touch_down:
                                touch_down = True
                                down_x = None
                                down_y = None
                                t_down = time.monotonic()

                elif ev_type == "EV_KEY" and "BTN_TOUCH" in ev_code:
                    if ev_val == "DOWN" and not touch_down:
                        touch_down = True
                        down_x = cur_x   # may be None; will latch on first ABS update
                        down_y = cur_y
                        t_down = time.monotonic()
                        print(f"[touch_capture] BTN_TOUCH DOWN  cur=({cur_x},{cur_y})")

                    elif ev_val == "UP" and touch_down:
                        touch_down = False
                        if cur_x is not None and cur_y is not None and t_down is not None:
                            swipe = Swipe(
                                down_px=(down_x or cur_x, down_y or cur_y),
                                up_px  =(cur_x, cur_y),
                                t_down =t_down,
                                t_up   =time.monotonic(),
                            )
                            with self._lock:
                                self._swipes.append(swipe)
                            self._event.set()
                            print(
                                f"[touch_capture] swipe via BTN_TOUCH: "
                                f"down={swipe.down_px} up={swipe.up_px}"
                            )
                        else:
                            print(
                                f"[touch_capture] BTN_TOUCH UP but coords incomplete: "
                                f"down=({down_x},{down_y}) cur=({cur_x},{cur_y})"
                            )
                        down_x = down_y = t_down = None

        except Exception as exc:
            print(f"[touch_capture] reader_loop error: {exc}")
