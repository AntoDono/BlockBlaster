"""Unified BlockBlaster launcher.

Usage
-----
Assist GUI (side-by-side viewer, works for both platforms)::

    uv run play.py --platform ios
    uv run play.py --platform android --mode assist

Android auto-play (fully hands-free)::

    uv run play.py --platform android
    uv run play.py --platform android --display        # with live preview window
    uv run play.py --platform android --serial emulator-5554

Controls (assist mode)
----------------------
    Tab          toggle calibration mode GRID / PIECES
    drag         set the bounding box for the active calibration mode
    R            clear the active mode's box
    D            dump per-slot debug images to assist_debug/
    Q / ESC      quit
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="play.py",
        description="BlockBlaster launcher — iOS assist or Android auto-play.",
    )
    p.add_argument(
        "--platform",
        choices=["ios", "android"],
        required=True,
        help="'ios' for the read-only assist GUI; 'android' for full auto-play.",
    )
    p.add_argument(
        "--mode",
        choices=["assist", "auto"],
        default=None,
        help=(
            "assist = side-by-side viewer (default for --platform ios); "
            "auto   = headless auto-player (default for --platform android, Android only)."
        ),
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Android auto mode only: open a pygame preview window.",
    )
    p.add_argument(
        "--serial",
        default=None,
        metavar="SERIAL",
        help="Android only: ADB device serial (auto-detected when omitted).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    platform: str = args.platform
    mode: str     = args.mode or ("auto" if platform == "android" else "assist")

    # ── iOS ───────────────────────────────────────────────────────────────────
    if platform == "ios":
        if mode == "auto":
            print(
                "[play] iOS auto-play is not supported.\n"
                "Apple does not expose a public touch-injection API.\n"
                "Use --platform android for hands-free play.",
                file=sys.stderr,
            )
            sys.exit(1)
        # assist mode — read-only viewer
        from blockblaster.assist.app import run
        run(platform="ios")
        return

    # ── Android ───────────────────────────────────────────────────────────────
    if mode == "assist":
        from blockblaster.assist.app import run
        from blockblaster.control.device import make_device
        device = make_device("android", serial=args.serial)
        run(device=device, platform="android")
        return

    if mode == "auto":
        if args.display:
            # Display path: open the full assist GUI (phone mirror + AI overlay)
            # and let it auto-execute moves on the device.
            from blockblaster.assist.app import run as assist_run
            from blockblaster.control.device import make_device
            device = make_device("android", serial=args.serial)
            assist_run(device=device, platform="android", auto_play=True)
            return
        # Headless path: bare auto-player, no window.
        from blockblaster.control.auto_player import run as auto_run
        auto_run(serial=args.serial, display=False)
        return

    print(f"[play] Unknown mode {mode!r}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
