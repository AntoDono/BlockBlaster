"""On-disk store for real (captured) piece crops.

Layout on disk::

    <root>/<piece_name>/<uuid>.png

Each image is a padded slot crop (the same representation the CNN sees).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np

from blockblaster.game.pieces import PIECES
from blockblaster.piece_cnn.config import INPUT_SIZE
from blockblaster.piece_cnn.synth import class_id_for

DEFAULT_DATA_DIR = Path("data/pieces")

_VALID_NAMES = {p.name for p in PIECES}
_PIECE_BY_NAME = {p.name: p for p in PIECES}


def dhash(img_bgr: np.ndarray, size: int = 8) -> int:
    """Return a 64-bit difference hash of a BGR image (scale/colour tolerant)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for value in diff.flatten():
        bits = (bits << 1) | int(value)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class RealPieceStore:
    """Saves labelled piece crops to disk under one folder per piece."""

    def __init__(self, root: str | Path = DEFAULT_DATA_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._counts: dict[str, int] = {}
        self.session_saved = 0
        self._reload()

    def _reload(self) -> None:
        self._counts = {
            d.name: sum(1 for _ in d.glob("*.png"))
            for d in self.root.iterdir() if d.is_dir()
        }

    def count(self, label: str) -> int:
        return self._counts.get(label, 0)

    def total(self) -> int:
        return sum(self._counts.values())

    def save(self, label: str, crop_bgr: np.ndarray) -> bool:
        """Save ``crop_bgr`` under ``label``; return ``True`` on success."""
        if label not in _VALID_NAMES:
            raise ValueError(f"unknown piece label: {label!r}")
        if crop_bgr is None or crop_bgr.size == 0:
            return False

        label_dir = self.root / label
        label_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(label_dir / f"{uuid.uuid4().hex}.png"), crop_bgr)
        self._counts[label] = self._counts.get(label, 0) + 1
        self.session_saved += 1
        return True


def load_real_dataset(
    root: str | Path = DEFAULT_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Load every saved crop as ``(images, labels)`` ready for training.

    Images are ``(N, INPUT_SIZE, INPUT_SIZE, 3)`` uint8 BGR; labels are int64
    class ids matching :func:`blockblaster.piece_cnn.synth.class_id_for`.
    """
    root = Path(root)
    imgs: list[np.ndarray] = []
    lbls: list[int] = []
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        piece = _PIECE_BY_NAME.get(label_dir.name)
        if piece is None:
            continue
        cid = class_id_for(piece)
        for png in label_dir.glob("*.png"):
            img = cv2.imread(str(png))
            if img is None:
                continue
            if img.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
                img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE),
                                 interpolation=cv2.INTER_AREA)
            imgs.append(img)
            lbls.append(cid)

    if not imgs:
        empty_imgs = np.empty((0, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        return empty_imgs, np.empty((0,), dtype=np.int64)
    return np.stack(imgs), np.asarray(lbls, dtype=np.int64)
