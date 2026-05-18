"""Shared ADB helpers used by every Android backend.

Centralises:
* Locating the ``adb`` binary (PATH first, then Android SDK install paths).
* Auto-detecting a single connected device (with BlueStacks TCP fallback).
* Running an ``adb`` command with retry on "error: closed".
* Reconnecting a dropped TCP serial.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

ADB_TIMEOUT = 8   # seconds for any single adb command

# Common locations where adb lives on macOS / Linux / Windows, in priority order.
_ADB_SEARCH_PATHS: list[Path] = [
    Path("adb"),
    Path.home() / "Library/Android/sdk/platform-tools/adb",
    Path.home() / "Android/Sdk/platform-tools/adb",
    *(
        Path(os.environ[v]) / "platform-tools/adb"
        for v in ("ANDROID_HOME", "ANDROID_SDK_ROOT")
        if v in os.environ
    ),
    Path("/usr/lib/android-sdk/platform-tools/adb"),
]


def _find_adb() -> str:
    """Return the path to a working adb binary, or raise RuntimeError."""
    for candidate in _ADB_SEARCH_PATHS:
        try:
            r = subprocess.run(
                [str(candidate), "version"], capture_output=True, timeout=4,
            )
            if r.returncode == 0:
                return str(candidate)
        except (FileNotFoundError, OSError):
            continue
    raise RuntimeError(
        "adb not found. Fix with one of:\n"
        "  • Add Android SDK platform-tools to PATH:\n"
        '      echo \'export PATH="$HOME/Library/Android/sdk/platform-tools:$PATH"\''
        " >> ~/.zshrc && source ~/.zshrc\n"
        "  • Or install via Homebrew: brew install android-platform-tools"
    )


# Resolve once at import time.
ADB_BIN: str = _find_adb()


def reconnect_tcp(serial: str) -> None:
    """Re-issue ``adb connect`` for TCP serials (BlueStacks, network)."""
    if not serial.startswith("127.") and ":" not in serial:
        return  # USB device — nothing to reconnect
    subprocess.run(
        [ADB_BIN, "connect", serial],
        capture_output=True, timeout=ADB_TIMEOUT,
    )


def adb_run(
    serial: str,
    args: list[str],
    *,
    retries: int = 2,
    text: bool = True,
) -> str:
    """Run ``adb -s {serial} {args}`` with retry on ``error: closed``."""
    for attempt in range(retries + 1):
        result = subprocess.run(
            [ADB_BIN, "-s", serial, *args],
            capture_output=True, text=text, timeout=ADB_TIMEOUT,
        )
        stderr = result.stderr if text else result.stderr.decode(errors="replace")
        if result.returncode == 0 and "error: closed" not in stderr:
            return result.stdout
        if "error: closed" in stderr and attempt < retries:
            print(
                f"[adb] connection closed (attempt {attempt + 1}/{retries + 1}),"
                f" reconnecting to {serial}…"
            )
            reconnect_tcp(serial)
            time.sleep(0.5 * (attempt + 1))
            continue
        raise RuntimeError(f"adb {' '.join(args)} failed: {stderr!r}")
    raise RuntimeError(f"adb {' '.join(args)} failed after {retries + 1} attempts")


def auto_detect_serial() -> str:
    """Return the serial of the only connected device, trying BlueStacks ports first."""
    def _list_devices() -> list[str]:
        r = subprocess.run(
            [ADB_BIN, "devices"], capture_output=True, text=True, timeout=ADB_TIMEOUT,
        )
        lines = [l for l in r.stdout.splitlines()[1:] if l.strip()]
        return [l.split()[0] for l in lines if "device" in l and "offline" not in l]

    devices = _list_devices()
    if not devices:
        for port in (5555, 5556, 5565, 5575):
            addr = f"127.0.0.1:{port}"
            subprocess.run(
                [ADB_BIN, "connect", addr], capture_output=True, timeout=ADB_TIMEOUT,
            )
            devices = _list_devices()
            if devices:
                print(f"[adb] Connected to BlueStacks at {addr}")
                break
    if not devices:
        raise RuntimeError(
            "No ADB device found.\n"
            "  • Android Studio AVD: start the emulator first.\n"
            "  • BlueStacks: enable ADB in Settings → Advanced → Android Debug Bridge,\n"
            "    then try running 'adb connect 127.0.0.1:5555' manually.\n"
            "  • USB phone: enable USB debugging, then run 'adb devices'."
        )
    if len(devices) > 1:
        raise RuntimeError(
            f"Multiple ADB devices found: {devices}.\n"
            "Pass --serial <serial> to pick one."
        )
    return devices[0]


def parse_wm_size(serial: str) -> tuple[int, int]:
    """Return ``(width_px, height_px)`` from ``adb shell wm size``.

    ``wm size`` may print both lines on Samsung devices with a display
    override (Settings → Display → Screen resolution)::

        Physical size: 1440x3088
        Override size: 1080x2220

    ``input swipe`` operates in the **override** coordinate space, so we
    prefer that line when present.  Falls back to "Physical size" otherwise.
    """
    raw = adb_run(serial, ["shell", "wm", "size"])
    physical: Optional[tuple[int, int]] = None
    override: Optional[tuple[int, int]] = None
    for line in raw.splitlines():
        low = line.lower()
        if "size:" not in low:
            continue
        parts = line.split(":")[-1].strip().split("x")
        if len(parts) != 2:
            continue
        try:
            wh = (int(parts[0]), int(parts[1]))
        except ValueError:
            continue
        if "override" in low:
            override = wh
        elif "physical" in low:
            physical = wh
        else:
            physical = physical or wh
    if override is not None:
        return override
    if physical is not None:
        return physical
    raise RuntimeError(f"Could not parse 'wm size' output: {raw!r}")


# Backwards-compat alias — touch_capture and other modules import ``_ADB_BIN``.
_ADB_BIN = ADB_BIN

__all__ = [
    "ADB_BIN",
    "ADB_TIMEOUT",
    "_ADB_BIN",
    "adb_run",
    "auto_detect_serial",
    "parse_wm_size",
    "reconnect_tcp",
]
