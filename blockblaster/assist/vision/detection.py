"""Whole-screen interactable-element detection for Block Blast.

Instead of hand-calibrating an 8x8 grid box, we scan the *entire* phone screen
and find the regions that stand out from the game's near-uniform background.
Block Blast renders almost everything (board, the three tray pieces, buttons,
score chrome) as bright/saturated shapes on a dark navy gradient, so a simple
"distance from background colour" segmentation isolates every interactable
blob.

Pipeline
--------
1. Estimate the background colour by averaging a small patch at the very bottom
   of the screen — that strip is always empty game background (dark navy).
2. Build a foreground mask of pixels that differ from the background by more
   than ``BG_DIFF_THRESHOLD`` (sum of absolute BGR channel diffs).
3. Morphologically close the mask so the gridded board cells and multi-cell
   pieces fuse into single solid blobs.
4. Run connected-components and keep blobs above a minimum area / size.

Run it directly to visualise what gets detected::

    uv run python -m blockblaster.assist.vision.detection --image shot.png
    uv run python -m blockblaster.assist.vision.detection --android [--serial SERIAL]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ── Tunables ──────────────────────────────────────────────────────────────────
BG_PATCH_H        = 5      # px; height of the bottom strip sampled for bg colour
BG_PATCH_W        = 100    # px; width (centred) of the bottom strip sampled for bg
BG_DIFF_THRESHOLD = 60     # min sum-of-abs BGR diff from bg to count as foreground
CLOSE_KERNEL      = 17     # px; fuses board cells & piece cells (incl. corner-
                           # touching S/Z/diagonal cells) into one solid blob
OPEN_KERNEL       = 3      # px; removes salt noise / single-pixel speckle
MIN_AREA_FRAC     = 0.0008 # min blob area as a fraction of the full frame
MIN_DIM_FRAC      = 0.02   # min blob width AND height as a fraction of frame dim


@dataclass
class Element:
    """A detected interactable region in original frame pixels."""

    x: int
    y: int
    w: int
    h: int
    area: int
    role: Optional[str] = None  # "board", "piece", or None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def fill_ratio(self) -> float:
        """Foreground pixels / bbox area — high for solid blocks, low for chrome."""
        bbox = self.w * self.h
        return self.area / bbox if bbox else 0.0


def estimate_background_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Return the background colour as a BGR int16 vector.

    The very bottom strip of the phone screen is always empty game background
    (below the piece tray), so we just average a small patch there. This is far
    more robust than modal-colour voting, which can latch onto the blue block
    colour when many cells are filled.
    """
    h, w = frame_bgr.shape[:2]
    cx   = w // 2
    half = min(BG_PATCH_W, w) // 2
    strip = frame_bgr[h - BG_PATCH_H:h, cx - half:cx + half]
    return strip.reshape(-1, 3).mean(axis=0).astype(np.int16)


def foreground_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Return a uint8 {0,255} mask of pixels that differ from the background."""
    bg   = estimate_background_bgr(frame_bgr)
    diff = np.abs(frame_bgr.astype(np.int16) - bg).sum(axis=2)
    mask = (diff > BG_DIFF_THRESHOLD).astype(np.uint8) * 255

    if OPEN_KERNEL > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_KERNEL, OPEN_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if CLOSE_KERNEL > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


CACHED_BOARD: Optional[Element] = None


def reset_board_cache() -> None:
    """Forget the cached board so the next detection re-finds it from scratch."""
    global CACHED_BOARD
    CACHED_BOARD = None


def detect_interactables(
    frame_bgr: np.ndarray,
    detect_board: bool = True,
) -> list[Element]:
    """Detect interactable blobs, largest first."""
    global CACHED_BOARD
    fh, fw = frame_bgr.shape[:2]
    mask   = foreground_mask(frame_bgr)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_area = MIN_AREA_FRAC * fw * fh
    min_w    = MIN_DIM_FRAC * fw
    min_h    = MIN_DIM_FRAC * fh

    elements: list[Element] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or w < min_w or h < min_h:
            continue
        elements.append(Element(int(x), int(y), int(w), int(h), int(area)))

    elements.sort(key=lambda e: e.area, reverse=True)
    _classify(elements)

    if elements and elements[0].role == "board":
        if detect_board or CACHED_BOARD is None:
            CACHED_BOARD = elements[0]
        else:
            for i, e in enumerate(elements):
                if e.role == "board":
                    elements[i] = CACHED_BOARD
                    break
            else:
                elements.insert(0, CACHED_BOARD)

    return elements


def _classify(elements: list[Element]) -> None:
    if not elements:
        return
    board = elements[0]
    board.role = "board"

    line_y = board.y + board.h
    below  = sorted(
        (e for e in elements[1:] if e.cy > line_y),
        key=lambda e: e.area, reverse=True,
    )
    for e in below[:3]:
        e.role = "piece"


def split_roles(
    elements: list[Element],
) -> tuple[Optional[Element], list[Element]]:
    """Return ``(board, pieces)`` where pieces are ordered left-to-right."""
    board = next((e for e in elements if e.role == "board"), None)
    pieces = sorted((e for e in elements if e.role == "piece"), key=lambda e: e.x)
    return board, pieces


def annotate(frame_bgr: np.ndarray, elements: list[Element]) -> np.ndarray:
    """Draw detected element boxes + labels onto a copy of the frame."""
    role_colors = {
        "board": (80, 240, 120),
        "piece": (0, 200, 255),
        None:    (0, 240, 220),
    }
    out = frame_bgr.copy()
    for idx, e in enumerate(elements):
        color = role_colors.get(e.role, role_colors[None])
        cv2.rectangle(out, (e.x, e.y), (e.x + e.w, e.y + e.h), color, 2)
        tag   = e.role if e.role else f"#{idx}"
        label = f"{tag} {e.w}x{e.h} f={e.fill_ratio:.2f}"
        ytxt  = e.y - 6 if e.y - 6 > 10 else e.y + 18
        cv2.putText(out, label, (e.x, ytxt), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)

    if elements:
        largest = elements[0]  # sorted largest-first
        y = min(largest.y + largest.h, out.shape[0] - 1)
        cv2.line(out, (0, y), (out.shape[1], y), (60, 80, 255), 2, cv2.LINE_AA)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_frame(args: argparse.Namespace) -> np.ndarray:
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read image: {args.image}")
        return frame

    from blockblaster.control.device import make_device

    device = make_device("android", serial=args.serial)
    device.start()
    try:
        import time
        for _ in range(100):
            frame, fid = device.get_latest_with_id()
            if frame is not None and fid > 0:
                return frame
            time.sleep(0.1)
        raise SystemExit("no frame captured from android device")
    finally:
        device.stop()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to a screenshot to analyse")
    src.add_argument("--android", action="store_true", help="capture from ADB device")
    p.add_argument("--serial", default=None, help="android ADB serial")
    p.add_argument("--save", default=None, help="write annotated output to this path")
    p.add_argument("--mask", action="store_true", help="also show the foreground mask")
    args = p.parse_args(argv)

    frame    = _load_frame(args)
    elements = detect_interactables(frame)
    print(f"detected {len(elements)} interactable element(s):")
    for idx, e in enumerate(elements):
        print(f"  #{idx}: role={e.role or '-':5s} x={e.x} y={e.y} w={e.w} h={e.h} "
              f"area={e.area} fill={e.fill_ratio:.2f}")

    out = annotate(frame, elements)
    if args.save:
        cv2.imwrite(args.save, out)
        print(f"saved annotated frame -> {args.save}")

    cv2.imshow("interactables", out)
    if args.mask:
        cv2.imshow("foreground mask", foreground_mask(frame))
    print("press any key in the image window to quit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
