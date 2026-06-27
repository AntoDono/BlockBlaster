"""Real piece-data collector — capture & label tray crops for the piece CNN.

Opens a GUI showing the live phone screen, the padded slot crops the CNN sees,
and a clickable piece palette. The CNN pre-labels each slot; fix any mistakes
and press ENTER to save. Duplicate captures of the same tray are skipped.

Usage::

    uv run collect.py --platform ios
    uv run collect.py --platform android [--serial <serial>]
    uv run collect.py --platform ios --out data/pieces
"""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collect.py",
        description="BlockBlaster real piece-data collector.",
    )
    p.add_argument("--platform", choices=["ios", "android"], required=True)
    p.add_argument(
        "--serial", default=None, metavar="SERIAL",
        help="Android only: ADB device serial (auto-detected when omitted).",
    )
    p.add_argument(
        "--out", default="data/pieces", metavar="DIR",
        help="Directory to write labelled crops into (default: data/pieces).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from blockblaster.assist.collector import run

    if args.platform == "ios":
        run(out_dir=args.out)
        return

    from blockblaster.control.device import make_device
    device = make_device("android", serial=args.serial)
    run(device=device, out_dir=args.out)


if __name__ == "__main__":
    main()
