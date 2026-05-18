"""Colour sampling helpers for the synthetic piece renderer."""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

from blockblaster.piece_cnn.config import (
    BG_HUE_RANGE,
    BG_JITTER,
    BG_SAT_RANGE,
    BG_VAL_RANGE,
    HSV_S_RANGE,
    HSV_V_RANGE,
    LOW_CONTRAST_DH,
    LOW_CONTRAST_DV,
    PER_CELL_HUE_JITTER,
    PER_CELL_S_JITTER,
    PER_CELL_V_JITTER,
)


def hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    hsv = np.array([[[h % 180, np.clip(s, 0, 255), np.clip(v, 0, 255)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def random_hsv(rng: random.Random) -> tuple[int, int, int]:
    return (
        rng.randint(0, 179),
        rng.randint(*HSV_S_RANGE),
        rng.randint(*HSV_V_RANGE),
    )


def per_cell_color(
    base_hsv: tuple[int, int, int],
    rng: random.Random,
    multi_color: bool,
) -> tuple[int, int, int]:
    """Return a BGR colour for one cell.

    If ``multi_color`` is True draw an independent random colour; otherwise
    apply a small HSV jitter around ``base_hsv`` so all cells look related.
    """
    if multi_color:
        return hsv_to_bgr(*random_hsv(rng))
    h, s, v = base_hsv
    h += rng.randint(-PER_CELL_HUE_JITTER, PER_CELL_HUE_JITTER)
    s += rng.randint(-PER_CELL_S_JITTER, PER_CELL_S_JITTER)
    v += rng.randint(-PER_CELL_V_JITTER, PER_CELL_V_JITTER)
    return hsv_to_bgr(h, s, v)


def scale_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return (
        int(np.clip(color[0] * factor, 0, 255)),
        int(np.clip(color[1] * factor, 0, 255)),
        int(np.clip(color[2] * factor, 0, 255)),
    )


def low_contrast_background(
    piece_hsv: tuple[int, int, int], rng: random.Random
) -> tuple[int, int, int]:
    """Sample a background HSV close to the piece HSV (low-contrast regime)."""
    h, _s, v = piece_hsv
    bg_h = (h + rng.randint(-LOW_CONTRAST_DH, LOW_CONTRAST_DH)) % 180
    bg_s = rng.randint(60, 220)
    dv   = rng.randint(-LOW_CONTRAST_DV, LOW_CONTRAST_DV)
    bg_v = int(np.clip(v + dv - 25, 15, 230))
    return bg_h, bg_s, bg_v


def random_background(
    h: int,
    w: int,
    rng: random.Random,
    forced_hsv: Optional[tuple[int, int, int]] = None,
) -> np.ndarray:
    """Solid background of a randomly-sampled colour, plus per-pixel noise and
    an optional brightness gradient or radial vignette.

    Pass ``forced_hsv`` to override the random base colour — used by the
    low-contrast regime to match the background tightly to the piece colour.
    """
    if forced_hsv is not None:
        base_h, base_s, base_v = forced_hsv
    else:
        base_h = rng.randint(*BG_HUE_RANGE)
        base_s = rng.randint(*BG_SAT_RANGE)
        base_v = rng.randint(*BG_VAL_RANGE)
    base_bgr = np.array(hsv_to_bgr(base_h, base_s, base_v), dtype=np.int16)

    bg    = np.full((h, w, 3), base_bgr, dtype=np.int16)
    noise = np.random.randint(-BG_JITTER, BG_JITTER + 1, (h, w, 3), dtype=np.int16)
    bg   += noise

    if rng.random() < 0.6:
        kind = rng.random()
        if kind < 0.5:
            grad = np.linspace(-15, 15, h, dtype=np.int16)[:, None, None]
            bg  += grad * rng.choice([-1, 1])
        else:
            cy, cx = h / 2, w / 2
            ys, xs = np.indices((h, w))
            dist   = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max(h, w)
            bg    += (15 - 30 * dist).astype(np.int16)[..., None]

    return np.clip(bg, 0, 255).astype(np.uint8)
