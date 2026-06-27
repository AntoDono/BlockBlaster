"""BlockBlaster launcher — assist GUI only.

Usage::

    uv run play.py --platform ios
    uv run play.py --platform android [--serial <serial>]
"""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="play.py",
        description="BlockBlaster assist GUI launcher.",
    )
    p.add_argument("--platform", choices=["ios", "android"], required=True)
    p.add_argument(
        "--serial", default=None, metavar="SERIAL",
        help="Android only: ADB device serial (auto-detected when omitted).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from blockblaster.assist.app import run

    if args.platform == "ios":
        run(platform="ios")
        return

    from blockblaster.control.device import make_device
    device = make_device("android", serial=args.serial)
    run(device=device, platform="android")


if __name__ == "__main__":
    main()
