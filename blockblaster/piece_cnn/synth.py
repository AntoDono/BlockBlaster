"""Synthetic Block Blast piece renderer for training the queue CNN.

Renders each canonical :class:`Piece` as a chamfered-cube grid on a
randomly-coloured background.  All rendering constants live in
:mod:`blockblaster.piece_cnn.config`, colour helpers in
:mod:`blockblaster.piece_cnn.color`, and drawing primitives in
:mod:`blockblaster.piece_cnn.draw`.
"""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

from blockblaster.game.pieces import PIECES, Piece
from blockblaster.piece_cnn.color import (
    low_contrast_background,
    per_cell_color,
    random_background,
    random_hsv,
)
from blockblaster.piece_cnn.config import (
    BORDER_BLACK_PROB,
    BORDER_FRAC_RANGE,
    EMPTY_CLASS_ID,
    GRADIENT_BEVEL_PROB,
    INPUT_SIZE,
    LOW_CONTRAST_PROB,
    MIN_CELL_PX,
    MULTI_COLOR_PROB,
    NUM_CLASSES,
    NUM_PIECES,
    PIECE_SIZE_FRAC_RANGE,
    ROTATION_JITTER_DEG,
    SHADOW_PROB,
    SHEAR_JITTER,
    SLOT_ASPECT_WH_RANGE,
    SLOT_HEIGHT_RANGE,
)
from blockblaster.piece_cnn.draw import (
    draw_cell,
    draw_piece_shadow,
    maybe_apply_color_cast,
    maybe_apply_jpeg,
    maybe_apply_occlusion,
)

# Re-export public constants so callers can do
#   from blockblaster.piece_cnn.synth import NUM_CLASSES
__all__ = [
    "EMPTY_CLASS_ID",
    "INPUT_SIZE",
    "NUM_CLASSES",
    "NUM_PIECES",
    "class_id_for",
    "generate_batch",
    "piece_for_class",
    "pregenerate_dataset",
    "render_piece_sample",
]


# ---------------------------------------------------------------------------
# Class-id helpers
# ---------------------------------------------------------------------------

def class_id_for(piece: Optional[Piece]) -> int:
    """Map a Piece (or None for empty) to its class id."""
    return EMPTY_CLASS_ID if piece is None else piece.piece_id


def piece_for_class(class_id: int) -> Optional[Piece]:
    """Inverse of :func:`class_id_for`."""
    if class_id == EMPTY_CLASS_ID:
        return None
    for p in PIECES:
        if p.piece_id == class_id:
            return p
    return None


# ---------------------------------------------------------------------------
# Sample renderer
# ---------------------------------------------------------------------------

def render_piece_sample(
    piece: Optional[Piece],
    rng: random.Random,
    slot_h: Optional[int] = None,
    slot_w: Optional[int] = None,
    cell_px: Optional[int] = None,
) -> np.ndarray:
    """Render one synthetic slot crop (BGR uint8, ``INPUT_SIZE × INPUT_SIZE``).

    Pass ``piece=None`` to render an empty slot.  Random parameters are drawn
    from the configured ranges unless overridden via keyword args.
    """
    if slot_h is None:
        slot_h = rng.randint(*SLOT_HEIGHT_RANGE)
    if slot_w is None:
        slot_w = max(40, int(round(slot_h * rng.uniform(*SLOT_ASPECT_WH_RANGE))))

    # Decide before drawing the background whether to enter the low-contrast
    # regime so the bg colour can be matched to the piece colour.
    base_hsv: Optional[tuple[int, int, int]] = None
    forced_bg_hsv: Optional[tuple[int, int, int]] = None
    if piece is not None:
        base_hsv = random_hsv(rng)
        if rng.random() < LOW_CONTRAST_PROB:
            forced_bg_hsv = low_contrast_background(base_hsv, rng)

    canvas = random_background(slot_h, slot_w, rng, forced_hsv=forced_bg_hsv)

    if piece is not None:
        assert base_hsv is not None
        if cell_px is None:
            target_frac = rng.uniform(*PIECE_SIZE_FRAC_RANGE)
            short_dim   = min(slot_w, slot_h)
            longest     = max(piece.rows, piece.cols)
            cell_px     = max(MIN_CELL_PX, int(target_frac * short_dim / longest))
        max_cell_w = max(MIN_CELL_PX, (slot_w - 8) // piece.cols)
        max_cell_h = max(MIN_CELL_PX, (slot_h - 8) // piece.rows)
        cell_px    = max(MIN_CELL_PX, min(cell_px, max_cell_w, max_cell_h))

        piece_w = piece.cols * cell_px
        piece_h = piece.rows * cell_px
        free_x  = max(0, slot_w - piece_w)
        free_y  = max(0, slot_h - piece_h)
        jx      = free_x // 4
        jy      = free_y // 4
        ox = free_x // 2 + (rng.randint(-jx, jx) if jx > 0 else 0)
        oy = free_y // 2 + (rng.randint(-jy, jy) if jy > 0 else 0)
        ox = int(np.clip(ox, 0, slot_w - piece_w))
        oy = int(np.clip(oy, 0, slot_h - piece_h))

        if rng.random() < SHADOW_PROB:
            draw_piece_shadow(canvas, piece, ox, oy, cell_px, rng)

        multi_color         = rng.random() < MULTI_COLOR_PROB
        use_gradient_bevel  = rng.random() < GRADIENT_BEVEL_PROB
        border_frac         = rng.uniform(*BORDER_FRAC_RANGE)
        use_black_border    = rng.random() < BORDER_BLACK_PROB

        for r, c in piece.cells:
            cell_color   = per_cell_color(base_hsv, rng, multi_color)
            border_color = (0, 0, 0) if use_black_border else None
            draw_cell(
                canvas,
                ox + c * cell_px,
                oy + r * cell_px,
                cell_px,
                cell_color,
                gradient_bevel=use_gradient_bevel,
                border_frac=border_frac,
                border_color=border_color,
            )

    if piece is not None and rng.random() < 0.7:
        angle = rng.uniform(-ROTATION_JITTER_DEG, ROTATION_JITTER_DEG)
        shear = rng.uniform(-SHEAR_JITTER, SHEAR_JITTER)
        ch, cw = canvas.shape[:2]
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
        M[0, 1] += shear
        M[1, 0] += shear * 0.5
        canvas = cv2.warpAffine(
            canvas, M, (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    if rng.random() < 0.4:
        k      = rng.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

    if rng.random() < 0.6:
        alpha  = 0.85 + rng.random() * 0.3
        beta   = -10 + rng.randint(0, 20)
        canvas = np.clip(canvas.astype(np.int16) * alpha + beta, 0, 255).astype(np.uint8)

    canvas = maybe_apply_color_cast(canvas, rng)
    maybe_apply_occlusion(canvas, rng)
    canvas = cv2.resize(canvas, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    canvas = maybe_apply_jpeg(canvas, rng)
    return canvas


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

def generate_batch(
    batch_size: int,
    rng: random.Random,
    empty_fraction: float = 1 / NUM_CLASSES,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (images, labels) on the fly.

    Images: ``(batch_size, INPUT_SIZE, INPUT_SIZE, 3)`` uint8 BGR.
    Labels: ``(batch_size,)`` int64 class ids in ``[0, NUM_CLASSES)``.
    """
    images = np.empty((batch_size, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    labels = np.empty((batch_size,), dtype=np.int64)
    for i in range(batch_size):
        piece     = None if rng.random() < empty_fraction else rng.choice(PIECES)
        images[i] = render_piece_sample(piece, rng)
        labels[i] = class_id_for(piece)
    return images, labels


# ---------------------------------------------------------------------------
# Parallel pre-generation
# ---------------------------------------------------------------------------

def _worker_chunk(args: tuple[int, int, float]) -> tuple[np.ndarray, np.ndarray]:
    seed, chunk_size, empty_fraction = args
    rng = random.Random(seed)
    return generate_batch(chunk_size, rng, empty_fraction=empty_fraction)


def pregenerate_dataset(
    n_samples: int,
    n_workers: int = 1,
    empty_fraction: float = 1 / NUM_CLASSES,
    chunk_size: int = 512,
    seed: int = 0,
    desc: str = "pregen",
) -> tuple[np.ndarray, np.ndarray]:
    """Render ``n_samples`` synthetic images in parallel and return them in RAM.

    Returns ``(images, labels)`` where images is uint8
    ``(N, INPUT_SIZE, INPUT_SIZE, 3)`` and labels is int64 ``(N,)``.
    """
    from tqdm import tqdm

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    tasks    = [
        (seed + i, min(chunk_size, n_samples - i * chunk_size), empty_fraction)
        for i in range(n_chunks)
    ]

    images  = np.empty((n_samples, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    labels  = np.empty((n_samples,), dtype=np.int64)
    written = 0

    if n_workers <= 1:
        iterator = (_worker_chunk(a) for a in tasks)
    else:
        from multiprocessing import get_context
        ctx      = get_context("spawn")
        pool     = ctx.Pool(processes=n_workers)
        iterator = pool.imap_unordered(_worker_chunk, tasks)

    try:
        with tqdm(total=n_samples, desc=desc, unit="img", leave=False) as bar:
            for imgs, lbls in iterator:
                n = len(imgs)
                images[written : written + n] = imgs
                labels[written : written + n] = lbls
                written += n
                bar.update(n)
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()

    return images, labels
