"""Tiny CNN classifier for the Block Blast queue (32 pieces + empty)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blockblaster.piece_cnn.synth import (
    INPUT_SIZE,
    NUM_CLASSES,
    piece_for_class,
)
from blockblaster.game.pieces import Piece

DEFAULT_WEIGHT_PATH = Path("piece_cnn.pt")


class PieceCNN(nn.Module):
    """Compact convnet — ~110 K params, runs in <1 ms per crop on CPU."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,  32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 96, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.bn2   = nn.BatchNorm2d(64)
        self.bn3   = nn.BatchNorm2d(96)
        self.pool  = nn.MaxPool2d(2)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Linear(96, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 64 → 32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 32 → 16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 16 →  8
        x = self.gap(x).flatten(1)                        # → (B, 96)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                                # logits


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

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
    if isinstance(imgs_bgr, np.ndarray) and imgs_bgr.ndim == 4:
        # Assumed (B, H, W, 3) uint8 BGR
        if imgs_bgr.shape[1:3] != (INPUT_SIZE, INPUT_SIZE):
            imgs_bgr = np.stack([
                cv2.resize(im, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
                for im in imgs_bgr
            ])
        rgb = imgs_bgr[..., ::-1].astype(np.float32) / 255.0  # BGR → RGB
        return torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
    return torch.stack([preprocess_bgr(im) for im in imgs_bgr])


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

class PieceClassifier:
    """Stateful wrapper that loads weights and classifies slot crops."""

    def __init__(
        self,
        weight_path: str | Path = DEFAULT_WEIGHT_PATH,
        device: str | torch.device = "cpu",
        confidence_threshold: float = 0.55,
    ) -> None:
        self.weight_path = Path(weight_path)
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
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
        """Classify a list of slot crops; return (piece_or_None, confidence) per slot."""
        if self.net is None or not slot_crops:
            return [(None, 0.0)] * len(slot_crops)
        batch = preprocess_batch(slot_crops).to(self.device)
        logits = self.net(batch)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        out: list[tuple[Optional[Piece], float]] = []
        for row in probs:
            cid = int(row.argmax())
            conf = float(row[cid])
            piece = piece_for_class(cid)
            if conf < self.confidence_threshold:
                out.append((None, conf))
            else:
                out.append((piece, conf))
        return out
