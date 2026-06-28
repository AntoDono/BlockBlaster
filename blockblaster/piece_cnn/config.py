"""Rendering constants and augmentation knobs for the synthetic piece renderer."""

from __future__ import annotations

from blockblaster.game.pieces import PIECES

# ── Class-id mapping ─────────────────────────────────────────────────────────
NUM_PIECES     = len(PIECES)
EMPTY_CLASS_ID = NUM_PIECES
NUM_CLASSES    = NUM_PIECES + 1
INPUT_SIZE     = 64

# ── Slot canvas geometry ──────────────────────────────────────────────────────
SLOT_HEIGHT_RANGE    = (220, 460)
SLOT_ASPECT_WH_RANGE = (0.55, 1.05)   # slot_w / slot_h

# Cell-first sizing: every cell uses the same pixel pitch regardless of shape,
# exactly like the real game. This is the only size signal the model has for
# 1x1 vs 2x1 or 4x1 vs 5x1, so never derive cell_px from a piece footprint.
CELL_PX_RANGE = (22, 58)
MIN_CELL_PX   = 14            # overflowing pieces grow the slot, not shrink the cell

# ── Background colour ─────────────────────────────────────────────────────────
BG_HUE_RANGE = (0, 179)
BG_SAT_RANGE = (0, 220)
BG_VAL_RANGE = (20, 210)
BG_JITTER    = 18

# ── Piece colour ──────────────────────────────────────────────────────────────
HSV_S_RANGE = (140, 240)
HSV_V_RANGE = (170, 240)

PER_CELL_HUE_JITTER = 6
PER_CELL_S_JITTER   = 30
PER_CELL_V_JITTER   = 25
MULTI_COLOR_PROB    = 0.20

# ── Geometric augmentation ────────────────────────────────────────────────────
ROTATION_JITTER_DEG = 7.0
SHEAR_JITTER        = 0.06

# ── Drop shadow ───────────────────────────────────────────────────────────────
SHADOW_PROB        = 0.9
SHADOW_OFFSET_FRAC = (0.05, 0.18)   # fraction of cell pitch
SHADOW_ALPHA_RANGE = (0.10, 0.28)
SHADOW_BLUR_FRAC   = (0.08, 0.22)

# ── Cell bevel / border ───────────────────────────────────────────────────────
GRADIENT_BEVEL_PROB  = 0.85
BORDER_FRAC_RANGE    = (0.04, 0.10)
BORDER_BLACK_PROB    = 0.25
BORDER_DARKEN_FACTOR = 0.45

# ── Low-contrast regime ───────────────────────────────────────────────────────
# Some samples get a background close in hue/value to the piece so the model
# cannot rely on colour contrast alone.
LOW_CONTRAST_PROB = 0.30
LOW_CONTRAST_DV   = 50              # max |V_piece − V_bg|
LOW_CONTRAST_DH   = 12             # max |H_piece − H_bg| (degrees)

# ── Photo-realistic corruption ────────────────────────────────────────────────
JPEG_PROB           = 0.4
JPEG_QUALITY_RANGE  = (45, 90)

COLOR_CAST_PROB = 0.35
COLOR_CAST_MAX  = 18

OCCLUSION_PROB      = 0.10
OCCLUSION_SIZE_FRAC = (0.05, 0.22)

# ── Per-piece sampling weights ───────────────────────────────────────────────
# Oversample historically-confused classes (long bars, 5-cell L-shapes).
# Keyed by piece name; anything not listed gets weight 1.0.
PIECE_SAMPLE_WEIGHTS: dict[str, float] = {
    "1x4": 3.0, "1x5": 3.0,
    "4x1": 3.0, "5x1": 3.0,
    "1x3": 1.5, "3x1": 1.5,
    "L_5_TL": 2.0, "L_5_TR": 2.0, "L_5_BL": 2.0, "L_5_BR": 2.0,
}

# ── Clean / game-view samples ────────────────────────────────────────────────
# Fraction of samples rendered with no photo-realistic corruption — matches the
# pristine in-game view the capture path feeds the CNN most of the time.
CLEAN_SAMPLE_FRACTION = 0.5

# Covers light pastels plus darker muted/earth tones (real captures hit V≈120).
CLEAN_BG_SAT_RANGE = (15, 150)
CLEAN_BG_VAL_RANGE = (110, 240)
CLEAN_LOW_CONTRAST_PROB = 0.15

CLEAN_BORDER_FRAC_RANGE = (0.012, 0.030)
CLEAN_BORDER_DARKEN     = 0.55

# Piece long-axis fill fraction of the slot. Wide range teaches multi-scale
# cell counting instead of overfitting to one pitch.
CLEAN_PIECE_FILL_RANGE = (0.40, 0.88)

CELL_CORNER_RADIUS_FRAC = 0.16
