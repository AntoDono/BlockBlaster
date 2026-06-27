"""Cell, shadow, and photo-corruption drawing primitives for the synth renderer."""

from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np

from blockblaster.game.pieces import Piece
from blockblaster.piece_cnn.color import scale_color
from blockblaster.piece_cnn.config import (
    BORDER_DARKEN_FACTOR,
    CELL_CORNER_RADIUS_FRAC,
    COLOR_CAST_MAX,
    COLOR_CAST_PROB,
    JPEG_PROB,
    JPEG_QUALITY_RANGE,
    OCCLUSION_PROB,
    OCCLUSION_SIZE_FRAC,
    SHADOW_ALPHA_RANGE,
    SHADOW_BLUR_FRAC,
    SHADOW_OFFSET_FRAC,
)


# ---------------------------------------------------------------------------
# Cell drawing
# ---------------------------------------------------------------------------

def _rounded_rect_mask(h: int, w: int, radius: int) -> np.ndarray:
    """Float32 alpha mask in [0, 1] for a rounded rectangle of size (h, w)."""
    radius = max(0, min(radius, h // 2, w // 2))
    if radius <= 0:
        return np.ones((h, w), dtype=np.float32)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    # Knock out the four corner squares
    mask[:radius, :radius]    = 0
    mask[:radius, w - radius:] = 0
    mask[h - radius:, :radius] = 0
    mask[h - radius:, w - radius:] = 0
    # Re-fill each corner with an anti-aliased quarter-disc
    for cy_corner, cx_corner, ys, xs in (
        (radius, radius, slice(0, radius), slice(0, radius)),
        (radius, w - radius, slice(0, radius), slice(w - radius, w)),
        (h - radius, radius, slice(h - radius, h), slice(0, radius)),
        (h - radius, w - radius, slice(h - radius, h), slice(w - radius, w)),
    ):
        corner = np.zeros((radius, radius), dtype=np.uint8)
        # cv2.circle gives nice AA on the boundary
        rel_cy = cy_corner - ys.start
        rel_cx = cx_corner - xs.start
        cv2.circle(corner, (rel_cx, rel_cy), radius, 255, -1, lineType=cv2.LINE_AA)
        mask[ys, xs] = np.maximum(mask[ys, xs], corner)
    return mask.astype(np.float32) / 255.0


def draw_cell(
    canvas: np.ndarray,
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
    *,
    gradient_bevel: bool = False,
    border_frac: float = 0.07,
    border_color: Optional[tuple[int, int, int]] = None,
    rounded: bool = False,
) -> None:
    """Draw one cell at pixel (x, y).

    ``gradient_bevel`` uses a smooth top→bottom value ramp; ``border_color``
    defaults to a darker shade of ``color``; ``rounded`` alpha-composites the
    cell so the background shows through the corner cut-outs.
    """
    if size < 4:
        return
    border_min = 1 if border_frac < 0.04 else 2
    border     = max(border_min, int(round(size * border_frac)))
    inset   = border
    fill_x0 = x + inset
    fill_y0 = y + inset
    fill_x1 = x + size - inset - 1
    fill_y1 = y + size - inset - 1

    if rounded:
        # Build the cell in a local tile, then alpha-blit through a rounded mask.
        tile_h = tile_w = size
        if border_color is None:
            border_color = scale_color(color, BORDER_DARKEN_FACTOR)
        tile = np.empty((tile_h, tile_w, 3), dtype=np.uint8)
        tile[:] = border_color
        fh = max(1, tile_h - 2 * inset)
        fw = max(1, tile_w - 2 * inset)
        if gradient_bevel:
            light = np.array(scale_color(color, 1.18), dtype=np.float32)
            dark  = np.array(scale_color(color, 0.72), dtype=np.float32)
            ramp  = np.linspace(0.0, 1.0, fh, dtype=np.float32)[:, None]
            col   = (1.0 - ramp) * light + ramp * dark
            block = np.broadcast_to(col[:, None, :], (fh, fw, 3)).astype(np.uint8)
            tile[inset:inset + fh, inset:inset + fw] = block
        else:
            tile[inset:inset + fh, inset:inset + fw] = color

        radius = int(round(size * CELL_CORNER_RADIUS_FRAC))
        mask   = _rounded_rect_mask(tile_h, tile_w, radius)[..., None]

        ch, cw = canvas.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(cw, x + tile_w), min(ch, y + tile_h)
        if x1 <= x0 or y1 <= y0:
            return
        tx0, ty0 = x0 - x, y0 - y
        tx1, ty1 = tx0 + (x1 - x0), ty0 + (y1 - y0)
        sub_canvas = canvas[y0:y1, x0:x1].astype(np.float32)
        sub_tile   = tile[ty0:ty1, tx0:tx1].astype(np.float32)
        sub_mask   = mask[ty0:ty1, tx0:tx1]
        blended    = sub_tile * sub_mask + sub_canvas * (1.0 - sub_mask)
        canvas[y0:y1, x0:x1] = blended.astype(np.uint8)
        return

    if gradient_bevel:
        fh    = max(1, fill_y1 - fill_y0)
        fw    = max(1, fill_x1 - fill_x0)
        light = np.array(scale_color(color, 1.25), dtype=np.float32)
        dark  = np.array(scale_color(color, 0.62), dtype=np.float32)
        ramp  = np.linspace(0.0, 1.0, fh, dtype=np.float32)[:, None]
        col   = (1.0 - ramp) * light + ramp * dark
        block = np.broadcast_to(col[:, None, :], (fh, fw, 3)).astype(np.uint8)
        canvas[fill_y0:fill_y0 + fh, fill_x0:fill_x0 + fw] = block
    else:
        cv2.rectangle(canvas, (fill_x0, fill_y0), (fill_x1, fill_y1), color, -1)
        light = scale_color(color, 1.35)
        dark  = scale_color(color, 0.55)
        bevel = max(1, size // 10)
        cv2.line(canvas, (fill_x0, fill_y0), (fill_x1, fill_y0), light, bevel)
        cv2.line(canvas, (fill_x0, fill_y0), (fill_x0, fill_y1), light, bevel)
        cv2.line(canvas, (fill_x0, fill_y1), (fill_x1, fill_y1), dark,  bevel)
        cv2.line(canvas, (fill_x1, fill_y0), (fill_x1, fill_y1), dark,  bevel)

    if border_color is None:
        border_color = scale_color(color, BORDER_DARKEN_FACTOR)
    cv2.rectangle(
        canvas,
        (x, y),
        (x + size - 1, y + size - 1),
        border_color,
        border,
    )


# ---------------------------------------------------------------------------
# Drop shadow
# ---------------------------------------------------------------------------

def draw_piece_shadow(
    canvas: np.ndarray,
    piece: Piece,
    ox: int,
    oy: int,
    cell_px: int,
    rng: random.Random,
) -> None:
    """Composite a soft drop-shadow of the piece onto ``canvas`` in-place."""
    h, w      = canvas.shape[:2]
    dx        = int(round(cell_px * rng.uniform(*SHADOW_OFFSET_FRAC)))
    dy        = int(round(cell_px * rng.uniform(*SHADOW_OFFSET_FRAC)))
    alpha_pk  = rng.uniform(*SHADOW_ALPHA_RANGE)

    alpha = np.zeros((h, w), dtype=np.float32)
    for r, c in piece.cells:
        x0, y0 = ox + c * cell_px + dx, oy + r * cell_px + dy
        x1, y1 = x0 + cell_px, y0 + cell_px
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(w, x1), min(h, y1)
        if x1c > x0c and y1c > y0c:
            alpha[y0c:y1c, x0c:x1c] = alpha_pk

    if alpha.max() == 0:
        return

    sigma = max(0.5, cell_px * rng.uniform(*SHADOW_BLUR_FRAC))
    k     = int(sigma * 4) | 1
    alpha = cv2.GaussianBlur(alpha, (k, k), sigma)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    canvas[:] = (canvas.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Photo-realistic corruptions
# ---------------------------------------------------------------------------

def maybe_apply_color_cast(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply a small per-channel BGR offset (white-balance / screen tint)."""
    if rng.random() >= COLOR_CAST_PROB:
        return canvas
    cast = np.array(
        [rng.randint(-COLOR_CAST_MAX, COLOR_CAST_MAX) for _ in range(3)],
        dtype=np.int16,
    )
    return np.clip(canvas.astype(np.int16) + cast, 0, 255).astype(np.uint8)


def maybe_apply_occlusion(canvas: np.ndarray, rng: random.Random) -> None:
    """Optionally darken a small strip at one edge of the slot."""
    if rng.random() >= OCCLUSION_PROB:
        return
    h, w  = canvas.shape[:2]
    edge  = rng.choice(("top", "bottom", "left", "right"))
    frac  = rng.uniform(*OCCLUSION_SIZE_FRAC)
    color = np.array(
        [rng.randint(0, 60), rng.randint(0, 60), rng.randint(0, 60)],
        dtype=np.uint8,
    )
    alpha = rng.uniform(0.4, 0.9)
    if edge == "top":
        sl: tuple = (slice(0, max(1, int(h * frac))), slice(None))
    elif edge == "bottom":
        sl = (slice(h - max(1, int(h * frac)), h), slice(None))
    elif edge == "left":
        sl = (slice(None), slice(0, max(1, int(w * frac))))
    else:
        sl = (slice(None), slice(w - max(1, int(w * frac)), w))
    region     = canvas[sl].astype(np.float32)
    canvas[sl] = (region * (1.0 - alpha) + color * alpha).astype(np.uint8)


def maybe_apply_jpeg(canvas: np.ndarray, rng: random.Random) -> np.ndarray:
    """Round-trip through JPEG to inject realistic compression artefacts."""
    if rng.random() >= JPEG_PROB:
        return canvas
    q  = rng.randint(*JPEG_QUALITY_RANGE)
    ok, buf = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return canvas
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
