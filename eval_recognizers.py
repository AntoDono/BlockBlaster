"""Compare geometric (piece_mask) vs CNN piece recognition on labeled data/."""

from __future__ import annotations

import glob
import os
from collections import defaultdict

import cv2

from blockblaster.assist.vision.piece_recognizer import PieceRecognizer
from blockblaster.piece_cnn import PieceClassifier


def main() -> None:
    files = sorted(glob.glob("data/pieces/*/*.png"))
    rec = PieceRecognizer()
    cnn = PieceClassifier()
    print(f"images: {len(files)} | cnn ready: {cnn.is_ready}")

    disagree_on_cnn_fail: list[tuple[str, str, str]] = []
    bars = {"1x4", "1x5", "4x1", "5x1"}
    bar_geo_ok = bar_cnn_ok = bar_n = 0

    geo_ok = cnn_ok = 0
    geo_none = cnn_none = 0
    geo_size_err = cnn_size_err = 0
    geo_wrong: dict[tuple[str, str], int] = defaultdict(int)
    cnn_wrong: dict[tuple[str, str], int] = defaultdict(int)

    for f in files:
        label = os.path.basename(os.path.dirname(f))
        crop = cv2.imread(f)
        if crop is None:
            continue

        g_piece, _ = rec._recognize_one(crop)
        (c_piece, _), = cnn.classify_slots([crop])

        g_name = g_piece.name if g_piece else None
        c_name = c_piece.name if c_piece else None

        if c_name != label:
            disagree_on_cnn_fail.append((label, str(c_name), str(g_name)))
            print(f"CNN FAIL: {f}  label={label} cnn={c_name} geo={g_name}")

        if label in bars:
            bar_n += 1
            bar_geo_ok += int(g_name == label)
            bar_cnn_ok += int(c_name == label)

        if g_name == label:
            geo_ok += 1
        elif g_name is None:
            geo_none += 1
            geo_wrong[(label, "NONE")] += 1
        else:
            geo_wrong[(label, g_name)] += 1
            if _dims(label) != _dims(g_name):
                geo_size_err += 1

        if c_name == label:
            cnn_ok += 1
        elif c_name is None:
            cnn_none += 1
            cnn_wrong[(label, "NONE")] += 1
        else:
            cnn_wrong[(label, c_name)] += 1
            if _dims(label) != _dims(c_name):
                cnn_size_err += 1

    n = len(files)
    print("\n=== ACCURACY ===")
    print(f"geometric : {geo_ok}/{n} = {geo_ok / n:.1%}  "
          f"(none: {geo_none}, size-mismatch errs: {geo_size_err})")
    print(f"cnn       : {cnn_ok}/{n} = {cnn_ok / n:.1%}  "
          f"(none: {cnn_none}, size-mismatch errs: {cnn_size_err})")

    print("\n=== GEOMETRIC top confusions (label -> pred) ===")
    for (lbl, pred), k in sorted(geo_wrong.items(), key=lambda t: -t[1])[:20]:
        print(f"  {lbl:>8} -> {pred:<8} x{k}")
    print("\n=== CNN top confusions (label -> pred) ===")
    for (lbl, pred), k in sorted(cnn_wrong.items(), key=lambda t: -t[1])[:20]:
        print(f"  {lbl:>8} -> {pred:<8} x{k}")

    print("\n=== On images where CNN was WRONG: what did geometric say? ===")
    print(f"  {'label':>8} | {'cnn':<8} | geometric")
    for label, c_name, g_name in disagree_on_cnn_fail:
        flag = " <-- geo correct" if g_name == label else ""
        print(f"  {label:>8} | {c_name:<8} | {g_name}{flag}")

    print("\n=== Long-bar subset (1x4/1x5/4x1/5x1) accuracy ===")
    print(f"  geometric: {bar_geo_ok}/{bar_n} = {bar_geo_ok / bar_n:.1%}")
    print(f"  cnn      : {bar_cnn_ok}/{bar_n} = {bar_cnn_ok / bar_n:.1%}")


def _dims(name: str) -> tuple[int, int] | None:
    """Bounding-box dims for a piece name, via its canonical grid."""
    from blockblaster.game.pieces import PIECES
    for p in PIECES:
        if p.name == name:
            return (p.rows, p.cols)
    return None


if __name__ == "__main__":
    main()
