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
    CELL_PX_RANGE,
    CLEAN_BG_SAT_RANGE,
    CLEAN_BG_VAL_RANGE,
    CLEAN_BORDER_FRAC_RANGE,
    CLEAN_LOW_CONTRAST_PROB,
    CLEAN_PIECE_FILL_RANGE,
    CLEAN_SAMPLE_FRACTION,
    EMPTY_CLASS_ID,
    GRADIENT_BEVEL_PROB,
    INPUT_SIZE,
    LOW_CONTRAST_PROB,
    MIN_CELL_PX,
    MULTI_COLOR_PROB,
    NUM_CLASSES,
    NUM_PIECES,
    PIECE_SAMPLE_WEIGHTS,
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
    clean: bool = False,
) -> np.ndarray:
    """Render one synthetic slot crop (BGR uint8, ``INPUT_SIZE × INPUT_SIZE``).

    Pass ``piece=None`` to render an empty slot.  Random parameters are drawn
    from the configured ranges unless overridden via keyword args.

    ``clean=True`` produces a pristine in-game view: no perspective warp /
    blur / contrast jitter / colour cast / occlusion / JPEG, a single base
    colour per piece, and a normal-contrast background. Use this to model the
    bulk of real screen captures where the queue slot is shown verbatim.
    """
    if slot_h is None:
        slot_h = rng.randint(*SLOT_HEIGHT_RANGE)
    if slot_w is None:
        slot_w = max(40, int(round(slot_h * rng.uniform(*SLOT_ASPECT_WH_RANGE))))

    # Cell-first sizing: pick the per-cell pixel pitch FIRST, identical for
    # every piece shape (this is what the real game does). The piece footprint
    # then follows from cols/rows. If it would overflow the slot, grow the
    # slot rather than shrinking the cell, so a 1x1 cell stays the same size
    # as a single cell of a 5x1.
    if piece is not None and cell_px is None:
        cell_px = rng.randint(*CELL_PX_RANGE)
    if piece is not None:
        cell_px = max(MIN_CELL_PX, int(cell_px))

        if clean:
            # Size the slot so the piece's LONG axis fills `fill_frac` of the
            # matching slot dimension (height for tall pieces, width for wide
            # ones). Earlier we sized against the short dim, which let a 5x1
            # render as just 30% of slot height — after the 96x96 resize each
            # cell was ~6 px tall and the model literally could not tell 4 vs
            # 5 cells apart. This keeps cells big enough to count.
            fill_frac  = rng.uniform(*CLEAN_PIECE_FILL_RANGE)
            piece_w_px = piece.cols * cell_px
            piece_h_px = piece.rows * cell_px

            if piece.rows >= piece.cols:
                # Tall (or square) piece — match height.
                slot_h = max(piece_h_px + 8, int(round(piece_h_px / fill_frac)))
                aspect = rng.uniform(*SLOT_ASPECT_WH_RANGE)   # w/h
                slot_w = max(piece_w_px + 8, int(round(slot_h * aspect)))
            else:
                # Wide piece — match width.
                slot_w = max(piece_w_px + 8, int(round(piece_w_px / fill_frac)))
                aspect = rng.uniform(*SLOT_ASPECT_WH_RANGE)   # w/h
                slot_h = max(piece_h_px + 8, int(round(slot_w / aspect)))

        min_slot_w = piece.cols * cell_px + 8
        min_slot_h = piece.rows * cell_px + 8
        if slot_w < min_slot_w:
            slot_w = min_slot_w
        if slot_h < min_slot_h:
            slot_h = min_slot_h

    base_hsv: Optional[tuple[int, int, int]] = None
    forced_bg_hsv: Optional[tuple[int, int, int]] = None
    if piece is not None:
        base_hsv = random_hsv(rng)
        lc_prob  = CLEAN_LOW_CONTRAST_PROB if clean else LOW_CONTRAST_PROB
        if rng.random() < lc_prob:
            forced_bg_hsv = low_contrast_background(base_hsv, rng)

    if clean:
        canvas = random_background(
            slot_h, slot_w, rng,
            forced_hsv=forced_bg_hsv,
            sat_range=CLEAN_BG_SAT_RANGE,
            val_range=CLEAN_BG_VAL_RANGE,
            add_gradient=False,
            jitter=0,
        )
    else:
        canvas = random_background(slot_h, slot_w, rng, forced_hsv=forced_bg_hsv)

    if piece is not None:
        assert base_hsv is not None
        assert cell_px is not None
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

        multi_color         = (not clean) and rng.random() < MULTI_COLOR_PROB
        use_gradient_bevel  = True if clean else rng.random() < GRADIENT_BEVEL_PROB
        if clean:
            border_frac = rng.uniform(*CLEAN_BORDER_FRAC_RANGE)
        else:
            border_frac = rng.uniform(*BORDER_FRAC_RANGE)
        use_black_border    = (not clean) and rng.random() < BORDER_BLACK_PROB

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
                rounded=clean,
            )

    if not clean and piece is not None and rng.random() < 0.7:
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

    if not clean and rng.random() < 0.4:
        k      = rng.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

    if not clean and rng.random() < 0.6:
        alpha  = 0.85 + rng.random() * 0.3
        beta   = -10 + rng.randint(0, 20)
        canvas = np.clip(canvas.astype(np.int16) * alpha + beta, 0, 255).astype(np.uint8)

    if not clean:
        canvas = maybe_apply_color_cast(canvas, rng)
        maybe_apply_occlusion(canvas, rng)
    canvas = cv2.resize(canvas, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    if not clean:
        canvas = maybe_apply_jpeg(canvas, rng)
    return canvas


# ---------------------------------------------------------------------------
# Batch generator
# ---------------------------------------------------------------------------

_PIECE_WEIGHTS = [PIECE_SAMPLE_WEIGHTS.get(p.name, 1.0) for p in PIECES]


def generate_batch(
    batch_size: int,
    rng: random.Random,
    empty_fraction: float = 1 / NUM_CLASSES,
    clean_fraction: float = CLEAN_SAMPLE_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (images, labels) on the fly.

    Images: ``(batch_size, INPUT_SIZE, INPUT_SIZE, 3)`` uint8 BGR.
    Labels: ``(batch_size,)`` int64 class ids in ``[0, NUM_CLASSES)``.

    ``clean_fraction`` is the share of samples rendered with no photo-
    realistic corruption (see ``render_piece_sample`` ``clean=True``).
    Pieces are drawn according to ``PIECE_SAMPLE_WEIGHTS`` so historically-
    confused classes (long bars, 5-cell L's) get oversampled.
    """
    images = np.empty((batch_size, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    labels = np.empty((batch_size,), dtype=np.int64)
    for i in range(batch_size):
        if rng.random() < empty_fraction:
            piece = None
        else:
            piece = rng.choices(PIECES, weights=_PIECE_WEIGHTS, k=1)[0]
        clean     = rng.random() < clean_fraction
        images[i] = render_piece_sample(piece, rng, clean=clean)
        labels[i] = class_id_for(piece)
    return images, labels


# ---------------------------------------------------------------------------
# Parallel pre-generation
# ---------------------------------------------------------------------------

def _worker_chunk(args: tuple[int, int, float, float]) -> tuple[np.ndarray, np.ndarray]:
    seed, chunk_size, empty_fraction, clean_fraction = args
    rng = random.Random(seed)
    return generate_batch(
        chunk_size, rng,
        empty_fraction=empty_fraction,
        clean_fraction=clean_fraction,
    )


def pregenerate_dataset(
    n_samples: int,
    n_workers: int = 1,
    empty_fraction: float = 1 / NUM_CLASSES,
    clean_fraction: float = CLEAN_SAMPLE_FRACTION,
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
        (seed + i, min(chunk_size, n_samples - i * chunk_size),
         empty_fraction, clean_fraction)
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
