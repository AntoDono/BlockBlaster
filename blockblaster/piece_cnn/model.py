"""Tiny CNN classifier for the Block Blast queue (all pieces + empty)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blockblaster.piece_cnn.synth import (
    EMPTY_CLASS_ID,
    INPUT_SIZE,
    NUM_CLASSES,
    piece_for_class,
)
from blockblaster.game.pieces import Piece

DEFAULT_WEIGHT_PATH = Path("piece_cnn.pt")


class PieceCNN(nn.Module):
    """Shallow, resolution-preserving classifier.

    Counting cells (1x4 vs 1x5, L-shapes vs bars) depends on the thin gaps
    *between* cells. Pooling/averaging destroys that signal, so this net never
    downsamples: two stride-1 conv layers extract local edge/gap features at
    full ``INPUT_SIZE`` resolution, then a single linear head reads the whole
    spatial map. Simpler and far more accurate at counting than a deep, pooled
    stack.
    """

    def __init__(
        self, num_classes: int = NUM_CLASSES, size: int = INPUT_SIZE
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,  16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.dropout = nn.Dropout(0.2)
        self.fc      = nn.Linear(16 * size * size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout(x.flatten(1))
        return self.fc(x)


def preprocess_bgr(img_bgr: np.ndarray) -> torch.Tensor:
    """Convert a BGR uint8 image to the model's input tensor (C, H, W) float32."""
    if img_bgr.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        img_bgr = cv2.resize(
            img_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA
        )
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1))  # CHW


def preprocess_batch(imgs_bgr: list[np.ndarray] | np.ndarray) -> torch.Tensor:
    """Stack a list/array of BGR uint8 images into a (B, C, H, W) float32 tensor."""
    if isinstance(imgs_bgr, np.ndarray) and imgs_bgr.ndim == 4:  # (B, H, W, 3) BGR
        if imgs_bgr.shape[1:3] != (INPUT_SIZE, INPUT_SIZE):
            imgs_bgr = np.stack([
                cv2.resize(im, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
                for im in imgs_bgr
            ])
        rgb = imgs_bgr[..., ::-1].astype(np.float32) / 255.0  # BGR → RGB
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
    return torch.stack([preprocess_bgr(im) for im in imgs_bgr])


class PieceClassifier:
    """Stateful wrapper that loads weights and classifies slot crops."""

    def __init__(
        self,
        weight_path: str | Path = DEFAULT_WEIGHT_PATH,
        device: str | torch.device = "cpu",
        allow_empty: bool = False,
    ) -> None:
        self.weight_path = Path(weight_path)
        self.device = torch.device(device)
        # When False (default), the EMPTY class is never predicted: callers only
        # feed crops where a piece was already detected, so the CNN always
        # returns its best *piece* guess. Set True to let it report "empty".
        self.allow_empty = allow_empty
        self.net: Optional[PieceCNN] = None
        self.last_error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.weight_path.exists():
            self.last_error = f"piece CNN weights not found: {self.weight_path}"
            return
        try:
            net = PieceCNN()
            state = torch.load(self.weight_path, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            net.load_state_dict(state)
            net.to(self.device).eval()
            self.net = net
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"piece CNN load failed: {exc!r}"

    @property
    def is_ready(self) -> bool:
        return self.net is not None

    @torch.no_grad()
    def classify_slots(
        self,
        slot_crops: list[np.ndarray],
    ) -> list[tuple[Optional[Piece], float]]:
        """Classify slot crops; return ``(piece_or_None, confidence)`` per slot.

        Unless ``allow_empty`` is set, the EMPTY class is excluded from the
        argmax — callers only feed crops where a piece was already detected, so
        the CNN always returns its best *piece* guess. Confidence is still
        returned for downstream gating / closed-loop verification.
        """
        if self.net is None or not slot_crops:
            return [(None, 0.0)] * len(slot_crops)
        batch = preprocess_batch(slot_crops).to(self.device)
        logits = self.net(batch)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        if not self.allow_empty:
            probs[:, EMPTY_CLASS_ID] = -1.0  # a piece is present; never pick "empty"
        out: list[tuple[Optional[Piece], float]] = []
        for row in probs:
            cid = int(row.argmax())
            out.append((piece_for_class(cid), float(row[cid])))
        return out
