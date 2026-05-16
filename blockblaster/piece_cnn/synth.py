"""Synthetic Block Blast piece renderer for training the queue CNN.

Renders each canonical :class:`Piece` as a chamfered-cube grid (the
recognisable Block Blast aesthetic) on a dark navy background.  Cells are
drawn touching, with a thin black inter-cell border and lighter / darker
edges to fake 3-D shading.  The renderer accepts random colour, cell
pitch, position, and noise so the resulting samples cover the
distribution we expect to see at inference time.

The "empty" class (`piece=None`) renders just a noisy dark background so
the classifier can learn to say "no piece in this slot".
"""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

from blockblaster.game.pieces import PIECES, Piece

# Public constants
NUM_PIECES = len(PIECES)
EMPTY_CLASS_ID = NUM_PIECES            # class index for "empty slot"
NUM_CLASSES   = NUM_PIECES + 1
INPUT_SIZE    = 96                     # the CNN works on 96×96 RGB crops

# Default canvas sizes (slot dimensions before resize).  We render at
# higher resolution and rely on the resize-to-INPUT_SIZE step to bake in
# realistic anti-aliasing.
SLOT_W_RANGE  = (120, 260)
SLOT_H_RANGE  = (120, 260)
CELL_PX_RANGE = (16, 38)               # cell pitch in pixels

# Background colour — sampled per-image across a wide HSV range so the
# classifier doesn't lock onto "dark navy" as part of the piece signature.
# Real game backgrounds vary considerably (different levels, brightness,
# UI panel tints) so we cover dark→bright and any hue.
BG_HUE_RANGE = (0, 179)                # any colour
BG_SAT_RANGE = (0, 220)                # gray-ish to saturated
BG_VAL_RANGE = (20, 210)               # dark to medium-bright
BG_JITTER    = 18                      # per-channel noise on top of base

# Block Blast palette covers many saturated hues.  We sample HSV to get
# the same look for any colour the game might use.
HSV_S_RANGE = (140, 240)
HSV_V_RANGE = (170, 240)

# Per-cell colour variation — real Block Blast pieces are sometimes a
# uniform colour with subtle per-cell shading variation, and sometimes
# pieces are made up of cells in totally different colours (e.g. clear
# bonuses).  Match both regimes so the classifier doesn't lock onto
# colour uniformity as a feature.
PER_CELL_HUE_JITTER = 6                # ±degrees on H (0–179)
PER_CELL_S_JITTER   = 30               # ±points on S
PER_CELL_V_JITTER   = 25               # ±points on V
MULTI_COLOR_PROB    = 0.20             # chance every cell is independently coloured

# Geometric augmentation — small affine perturbations so the classifier
# is robust to rotation and minor perspective skew from the phone capture.
ROTATION_JITTER_DEG = 7.0              # ±degrees
SHEAR_JITTER        = 0.06             # ±shear factor


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _hsv_to_bgr(h: int, s: int, v: int) -> tuple[int, int, int]:
    hsv = np.array([[[h % 180, np.clip(s, 0, 255), np.clip(v, 0, 255)]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _random_hsv(rng: random.Random) -> tuple[int, int, int]:
    return (
        rng.randint(0, 179),
        rng.randint(*HSV_S_RANGE),
        rng.randint(*HSV_V_RANGE),
    )


def _random_piece_color(rng: random.Random) -> tuple[int, int, int]:
    """Sample a saturated piece colour in BGR."""
    return _hsv_to_bgr(*_random_hsv(rng))


def _per_cell_color(
    base_hsv: tuple[int, int, int],
    rng: random.Random,
    multi_color: bool,
) -> tuple[int, int, int]:
    """Return a BGR colour for one cell of a piece.

    If ``multi_color`` is True we draw an independent random colour;
    otherwise we apply a small HSV jitter around ``base_hsv`` so cells of
    the same piece look related but not identical.
    """
    if multi_color:
        return _hsv_to_bgr(*_random_hsv(rng))
    h, s, v = base_hsv
    h += rng.randint(-PER_CELL_HUE_JITTER, PER_CELL_HUE_JITTER)
    s += rng.randint(-PER_CELL_S_JITTER, PER_CELL_S_JITTER)
    v += rng.randint(-PER_CELL_V_JITTER, PER_CELL_V_JITTER)
    return _hsv_to_bgr(h, s, v)


def _scale_color(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return (
        int(np.clip(color[0] * factor, 0, 255)),
        int(np.clip(color[1] * factor, 0, 255)),
        int(np.clip(color[2] * factor, 0, 255)),
    )


def _random_background(
    h: int, w: int, rng: random.Random
) -> np.ndarray:
    """Solid background of a randomly-sampled colour, plus jitter / gradient.

    The base colour spans the full hue range and a wide brightness band so
    the classifier sees pieces on dark navy, light teal, mid-grey, even
    bright purple-ish backgrounds.  This keeps the model from coupling
    background colour to the "is a piece present?" decision.
    """
    base_h = rng.randint(*BG_HUE_RANGE)
    base_s = rng.randint(*BG_SAT_RANGE)
    base_v = rng.randint(*BG_VAL_RANGE)
    base_bgr = np.array(_hsv_to_bgr(base_h, base_s, base_v), dtype=np.int16)

    bg = np.full((h, w, 3), base_bgr, dtype=np.int16)
    # Per-pixel noise (uniform — fast and indistinguishable from gaussian
    # at this magnitude).
    noise = np.random.randint(-BG_JITTER, BG_JITTER + 1, (h, w, 3), dtype=np.int16)
    bg += noise

    # Optional brightness gradient (top→bottom or radial vignette).  Keeps
    # the strength small so the bg still reads as one colour.
    if rng.random() < 0.6:
        kind = rng.random()
        if kind < 0.5:
            # Linear vertical gradient
            grad = np.linspace(-15, 15, h, dtype=np.int16)[:, None, None]
            bg += grad * rng.choice([-1, 1])
        else:
            # Radial vignette (corners darker)
            cy, cx = h / 2, w / 2
            ys, xs = np.indices((h, w))
            dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max(h, w)
            bg += (15 - 30 * dist).astype(np.int16)[..., None]

    return np.clip(bg, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Cell + piece rendering
# ---------------------------------------------------------------------------

def _draw_cell(
    canvas: np.ndarray,
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a single chamfered cell at (x, y) on `canvas`."""
    if size < 4:
        return
    # Main fill (slightly inset so the dark border separates touching cells)
    cv2.rectangle(canvas, (x + 1, y + 1), (x + size - 2, y + size - 2), color, -1)

    light = _scale_color(color, 1.35)
    dark  = _scale_color(color, 0.55)

    bevel = max(1, size // 10)
    # Top + left highlight
    cv2.line(canvas, (x + 1, y + 1), (x + size - 2, y + 1), light, bevel)
    cv2.line(canvas, (x + 1, y + 1), (x + 1, y + size - 2), light, bevel)
    # Bottom + right shadow
    cv2.line(canvas, (x + 1, y + size - 2), (x + size - 2, y + size - 2), dark, bevel)
    cv2.line(canvas, (x + size - 2, y + 1), (x + size - 2, y + size - 2), dark, bevel)
    # Thin black outline → this is what creates the visible separator
    # between touching cells inside a piece.
    cv2.rectangle(canvas, (x, y), (x + size - 1, y + size - 1), (0, 0, 0), 1)


def render_piece_sample(
    piece: Optional[Piece],
    rng: random.Random,
    slot_h: Optional[int] = None,
    slot_w: Optional[int] = None,
    cell_px: Optional[int] = None,
) -> np.ndarray:
    """Render one synthetic slot crop (BGR uint8) of `INPUT_SIZE × INPUT_SIZE`.

    Pass `piece=None` to render an empty slot.  Random parameters are drawn
    from the configured ranges unless overridden via the keyword args.
    """
    if slot_h is None:
        slot_h = rng.randint(*SLOT_H_RANGE)
    if slot_w is None:
        slot_w = rng.randint(*SLOT_W_RANGE)

    canvas = _random_background(slot_h, slot_w, rng)

    if piece is not None:
        if cell_px is None:
            cell_px = rng.randint(*CELL_PX_RANGE)
        # Make sure the piece fits within the slot with some padding
        max_cell_w = max(4, (slot_w - 8) // piece.cols)
        max_cell_h = max(4, (slot_h - 8) // piece.rows)
        cell_px = min(cell_px, max_cell_w, max_cell_h)
        if cell_px < 6:
            cell_px = 6

        piece_w = piece.cols * cell_px
        piece_h = piece.rows * cell_px
        # Centre with random jitter (±15% of remaining padding)
        free_x = max(0, slot_w - piece_w)
        free_y = max(0, slot_h - piece_h)
        ox = free_x // 2 + rng.randint(-free_x // 4, free_x // 4)
        oy = free_y // 2 + rng.randint(-free_y // 4, free_y // 4)
        ox = int(np.clip(ox, 0, slot_w - piece_w))
        oy = int(np.clip(oy, 0, slot_h - piece_h))

        base_hsv = _random_hsv(rng)
        multi_color = rng.random() < MULTI_COLOR_PROB
        for r, c in piece.cells:
            cell_color = _per_cell_color(base_hsv, rng, multi_color)
            _draw_cell(
                canvas,
                ox + c * cell_px,
                oy + r * cell_px,
                cell_px,
                cell_color,
            )

    # Slight rotation + shear to mimic phone-capture skew (only when a
    # piece is present — avoids waste on blank backgrounds).
    if piece is not None and rng.random() < 0.7:
        angle = rng.uniform(-ROTATION_JITTER_DEG, ROTATION_JITTER_DEG)
        shear = rng.uniform(-SHEAR_JITTER, SHEAR_JITTER)
        ch, cw = canvas.shape[:2]
        M = cv2.getRotationMatrix2D((cw / 2, ch / 2), angle, 1.0)
        # Inject shear directly into the affine matrix
        M[0, 1] += shear
        M[1, 0] += shear * 0.5
        canvas = cv2.warpAffine(
            canvas, M, (cw, ch),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    # Optional gentle blur to mimic device downsampling
    if rng.random() < 0.4:
        k = rng.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

    # Optional brightness / contrast jitter
    if rng.random() < 0.6:
        alpha = 0.85 + rng.random() * 0.3      # contrast
        beta  = -10 + rng.randint(0, 20)        # brightness
        canvas = np.clip(canvas.astype(np.int16) * alpha + beta, 0, 255).astype(np.uint8)

    # Resize to model input
    canvas = cv2.resize(canvas, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return canvas


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
# Batch generator (used by the trainer)
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
        if rng.random() < empty_fraction:
            piece = None
        else:
            piece = rng.choice(PIECES)
        images[i] = render_piece_sample(piece, rng)
        labels[i] = class_id_for(piece)
    return images, labels


# ---------------------------------------------------------------------------
# Parallel pre-generation (used by the trainer for one-shot dataset build)
# ---------------------------------------------------------------------------

def _worker_chunk(args: tuple[int, int, float]) -> tuple[np.ndarray, np.ndarray]:
    """Worker entry: generate `chunk_size` samples with a deterministic seed."""
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
    """Render `n_samples` synthetic images in parallel and return them in RAM.

    Args:
        n_samples:      how many examples to generate in total.
        n_workers:      parallel processes (1 = serial, in-process).
        empty_fraction: fraction of samples that are empty slots.
        chunk_size:     samples per worker task; smaller = finer progress
                        updates, larger = lower IPC overhead.
        seed:           base seed for reproducibility (each chunk gets a
                        derived seed so workers don't produce identical data).
        desc:           label shown on the tqdm progress bar.

    Returns:
        (images, labels) where images is uint8 ``(N, INPUT_SIZE, INPUT_SIZE, 3)``
        and labels is int64 ``(N,)``.
    """
    from tqdm import tqdm

    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    tasks = [
        (seed + i, min(chunk_size, n_samples - i * chunk_size), empty_fraction)
        for i in range(n_chunks)
    ]

    images = np.empty((n_samples, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    labels = np.empty((n_samples,), dtype=np.int64)
    written = 0

    if n_workers <= 1:
        iterator = (_worker_chunk(args) for args in tasks)
    else:
        # Use spawn explicitly — safer than fork when CUDA is initialised.
        from multiprocessing import get_context
        ctx = get_context("spawn")
        pool = ctx.Pool(processes=n_workers)
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
