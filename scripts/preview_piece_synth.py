"""Dump a random batch of synthetic piece-CNN samples to disk for visual inspection.

Run BEFORE retraining the CNN to sanity-check that:

* every cell is drawn at the same pixel pitch regardless of piece shape
  (1x1 vs 2x1, 4x1 vs 5x1, etc. look obviously different)
* "clean" samples actually look like an in-game queue slot (no warp, blur,
  occlusion, JPEG, colour cast)
* "corrupt" samples still cover the photo-realistic regime

Usage:

    uv run scripts/preview_piece_synth.py

Output: ``tmp/piece_synth_samples/`` populated with ``{idx}__class{cid}_{name}__{mode}.png``
files. Sort the folder by name to group samples by class.
"""

from __future__ import annotations

import random
import shutil
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2

from blockblaster.game.pieces import PIECES
from blockblaster.piece_cnn.config import CLEAN_SAMPLE_FRACTION, NUM_CLASSES
from blockblaster.piece_cnn.synth import class_id_for, render_piece_sample

# ── CONFIG ────────────────────────────────────────────────────────────────────
NUM_SAMPLES   = 256
OUT_DIR       = Path("tmp/piece_synth_samples")
EMPTY_FRACTION = 1 / NUM_CLASSES
SEED          = 0


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    class_counts: Counter[int] = Counter()
    mode_counts: Counter[str]  = Counter()

    for i in range(NUM_SAMPLES):
        if rng.random() < EMPTY_FRACTION:
            piece = None
            name  = "empty"
        else:
            piece = rng.choice(PIECES)
            name  = piece.name.replace(" ", "_").replace("/", "_")

        clean = rng.random() < CLEAN_SAMPLE_FRACTION
        mode  = "clean" if clean else "corrupt"
        img   = render_piece_sample(piece, rng, clean=clean)

        cid = class_id_for(piece)
        fname = f"{i:04d}__class{cid:02d}_{name}__{mode}.png"
        cv2.imwrite(str(OUT_DIR / fname), img)

        class_counts[cid] += 1
        mode_counts[mode] += 1

    print(f"[preview] wrote {NUM_SAMPLES} samples to {OUT_DIR.resolve()}")
    print(f"[preview] modes: {dict(mode_counts)}")
    print(f"[preview] classes covered: {len(class_counts)} / {NUM_CLASSES}")
    print("[preview] per-class counts (class_id: n):")
    for cid in sorted(class_counts):
        print(f"   {cid:>2d}: {class_counts[cid]}")


if __name__ == "__main__":
    main()
