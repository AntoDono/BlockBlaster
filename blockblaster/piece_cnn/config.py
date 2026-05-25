"""Rendering constants and augmentation knobs for the synthetic piece renderer.

All magic numbers live here so they can be tuned without digging through
drawing code.
"""

from __future__ import annotations

from blockblaster.game.pieces import PIECES

# ── Class-id mapping ─────────────────────────────────────────────────────────
NUM_PIECES     = len(PIECES)
EMPTY_CLASS_ID = NUM_PIECES          # class index reserved for "empty slot"
NUM_CLASSES    = NUM_PIECES + 1
INPUT_SIZE     = 96                  # the CNN operates on 96×96 RGB crops

# ── Slot canvas geometry ──────────────────────────────────────────────────────
# Real Block Blast queue slots are taller than wide — measured ≈ 349×448
# (aspect w/h ≈ 0.78) in reference captures.  Render at higher resolution
# so the final downsample to INPUT_SIZE bakes in realistic anti-aliasing.
SLOT_HEIGHT_RANGE    = (220, 460)
SLOT_ASPECT_WH_RANGE = (0.55, 1.05)   # slot_w / slot_h

# Cell-first sizing: every cell is rendered at the same pixel pitch
# regardless of piece shape, exactly like the real game. This is the *only*
# size signal the model gets for distinguishing e.g. 1x1 vs 2x1 or 4x1 vs 5x1,
# so it must be enforced — never derive cell_px from a target piece footprint.
CELL_PX_RANGE = (22, 58)
MIN_CELL_PX   = 14            # hard floor; pieces that would overflow the slot
                              # grow the slot instead of shrinking the cell.

# ── Background colour ─────────────────────────────────────────────────────────
BG_HUE_RANGE = (0, 179)              # any hue
BG_SAT_RANGE = (0, 220)              # grey → saturated
BG_VAL_RANGE = (20, 210)             # dark → medium-bright
BG_JITTER    = 18                    # per-channel noise on top of base colour

# ── Piece colour ──────────────────────────────────────────────────────────────
HSV_S_RANGE = (140, 240)
HSV_V_RANGE = (170, 240)

# Per-cell HSV jitter (small jitter keeps cells of one piece looking related).
PER_CELL_HUE_JITTER = 6
PER_CELL_S_JITTER   = 30
PER_CELL_V_JITTER   = 25
MULTI_COLOR_PROB    = 0.20           # chance every cell gets an independent colour

# ── Geometric augmentation ────────────────────────────────────────────────────
ROTATION_JITTER_DEG = 7.0
SHEAR_JITTER        = 0.06

# ── Drop shadow ───────────────────────────────────────────────────────────────
# Real Block Blast pieces cast a subtle soft shadow to the bottom-right.
SHADOW_PROB        = 0.9
SHADOW_OFFSET_FRAC = (0.05, 0.18)   # offset as fraction of cell pitch
SHADOW_ALPHA_RANGE = (0.10, 0.28)
SHADOW_BLUR_FRAC   = (0.08, 0.22)

# ── Cell bevel style ─────────────────────────────────────────────────────────
# Real game uses a smooth top→bottom gradient; make this the dominant style.
GRADIENT_BEVEL_PROB = 0.85

# ── Cell border ───────────────────────────────────────────────────────────────
# Real game uses a thin border in a darker shade of the cell colour.
BORDER_FRAC_RANGE    = (0.04, 0.10)  # border thickness as fraction of cell pitch
BORDER_BLACK_PROB    = 0.25          # chance of pure-black border instead
BORDER_DARKEN_FACTOR = 0.45          # V multiplier for the coloured border

# ── Low-contrast regime ───────────────────────────────────────────────────────
# 30 % of piece samples get a background close in hue/value to the piece so
# the model cannot rely solely on colour contrast.
LOW_CONTRAST_PROB = 0.30
LOW_CONTRAST_DV   = 50              # max |V_piece − V_bg|
LOW_CONTRAST_DH   = 12             # max |H_piece − H_bg| (degrees)

# ── Photo-realistic corruption ────────────────────────────────────────────────
JPEG_PROB           = 0.4
JPEG_QUALITY_RANGE  = (45, 90)

COLOR_CAST_PROB = 0.35
COLOR_CAST_MAX  = 18               # ±BGR offset per channel

OCCLUSION_PROB      = 0.10
OCCLUSION_SIZE_FRAC = (0.05, 0.22)

# ── Per-piece sampling weights ───────────────────────────────────────────────
# Bias the random piece sampling so historically-confused classes get more
# training examples. Keyed by piece NAME (see blockblaster.game.pieces.PIECES).
# Anything not listed gets weight 1.0. Long bars and 5-cell L-shapes are the
# main repeat offenders for cell-miscounting; bump them ~3x.
PIECE_SAMPLE_WEIGHTS: dict[str, float] = {
    "1x4": 3.0, "1x5": 3.0,
    "4x1": 3.0, "5x1": 3.0,
    "1x3": 1.5, "3x1": 1.5,
    "L_5_TL": 2.0, "L_5_TR": 2.0, "L_5_BL": 2.0, "L_5_BR": 2.0,
}

# ── Clean / game-view samples ────────────────────────────────────────────────
# Fraction of generated samples rendered without ANY photo-realistic
# corruption (no warp/blur/contrast jitter/color cast/occlusion/jpeg, single
# base colour, normal-contrast background). Matches the pristine in-game view
# the OCR-style capture path actually feeds the CNN most of the time.
CLEAN_SAMPLE_FRACTION = 0.5

# Clean-mode background: covers light pastels AND darker muted/earth tones
# (olive, khaki, dusty rose, slate) — real captures hit V≈120 regularly, not
# just the bright-pastel range. Saturation goes up to 150 so we cover muted-
# but-not-grey tones.
CLEAN_BG_SAT_RANGE = (15, 150)
CLEAN_BG_VAL_RANGE = (110, 240)
# Allow occasional low-contrast bg in clean mode (real game sometimes shows a
# piece whose colour is close in value to the bg).
CLEAN_LOW_CONTRAST_PROB = 0.15

# Clean-mode cell border: thin, subtle, same hue as the cell (darker shade).
# In-game the inter-cell separator is barely visible — just enough to outline
# cells without dominating the look.
CLEAN_BORDER_FRAC_RANGE = (0.012, 0.030)
CLEAN_BORDER_DARKEN     = 0.55

# Clean-mode piece-to-slot fill: the rendered piece's long axis covers this
# fraction of the slot dimension matching that axis. Real captures vary a
# lot — sometimes the piece is tightly cropped, sometimes there's a lot of
# padding above/below it. A wide range here teaches the model to count
# cells at multiple scales instead of overfitting to one specific pitch.
CLEAN_PIECE_FILL_RANGE = (0.40, 0.88)

# Clean-mode rounded corners: cell corner radius as a fraction of cell pitch.
CELL_CORNER_RADIUS_FRAC = 0.16
