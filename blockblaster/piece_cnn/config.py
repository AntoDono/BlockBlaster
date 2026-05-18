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

# Fraction of the slot's *shorter* dimension the piece's longest axis fills.
# Real pieces sit in a lot of padding (~40–55% of the short side).
PIECE_SIZE_FRAC_RANGE = (0.30, 0.65)
MIN_CELL_PX           = 10            # never go below this even for large pieces

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
