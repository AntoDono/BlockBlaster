"""Phone-panel and calibration-overlay rendering for the assist GUI."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pygame

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.calibration import CalibrationBox

# ── Colours (imported by render.py for backward-compat re-export) ─────────────
PANEL_BG        = (20, 20, 30)
PANEL_BORDER    = (55, 55, 75)
TEXT_COLOR      = (220, 220, 235)
DIM_TEXT        = (110, 110, 130)
STATUS_BG       = (10, 10, 16)
DEVICE_OK_COL   = (60, 220, 100)
DEVICE_ERR_COL  = (220, 70, 70)
LABEL_COL       = (160, 160, 185)
OVERLAY_BOX     = (0, 240, 220)
OVERLAY_GRID    = (0, 180, 160)
OVERLAY_DOT     = (255, 255, 100)
DRAG_COLOR      = (255, 200, 0)
QUEUE_BOX_COLOR = (255, 210, 50)
QUEUE_DIVIDER   = (200, 165, 40)
SUGGEST_FILL    = (80, 240, 120)
SUGGEST_BORDER  = (40, 200, 80)
SUGGEST_FILL_A  = 170


def bgr_to_surface(
    frame_bgr: np.ndarray,
    target_rect: pygame.Rect,
) -> tuple[pygame.Surface, float, int, int]:
    """Convert a BGR frame to a pygame.Surface letterboxed into *target_rect*.

    Returns ``(surface, scale, blit_x, blit_y)``.
    """
    h, w   = frame_bgr.shape[:2]
    scale  = min(target_rect.width / w, target_rect.height / h)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb    = np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    surf   = pygame.image.frombuffer(rgb.tobytes(), (new_w, new_h), "RGB")
    blit_x = target_rect.x + (target_rect.width  - new_w) // 2
    blit_y = target_rect.y + (target_rect.height - new_h) // 2
    return surf, scale, blit_x, blit_y


def draw_phone_panel(
    screen: pygame.Surface,
    frame: Optional[np.ndarray],
    rect: pygame.Rect,
    error_msg: Optional[str],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    cached_surface: Optional[tuple[pygame.Surface, float, int, int]] = None,
) -> tuple[float, int, int]:
    """Draw the left phone-screen panel.  Returns ``(scale, blit_x, blit_y)``."""
    pygame.draw.rect(screen, PANEL_BG,     rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    lbl = small_font.render("PHONE SCREEN", True, LABEL_COL)
    screen.blit(lbl, (rect.x + 10, rect.y + 8))

    content_rect = pygame.Rect(rect.x + 4, rect.y + 30, rect.width - 8, rect.height - 38)

    if frame is not None:
        if cached_surface is not None:
            surf, scale, blit_x, blit_y = cached_surface
        else:
            surf, scale, blit_x, blit_y = bgr_to_surface(frame, content_rect)
        screen.blit(surf, (blit_x, blit_y))
        return scale, blit_x, blit_y

    msg   = error_msg if error_msg else "Waiting for device…"
    lines = _wrap(msg, small_font, content_rect.width - 20)
    total_h = len(lines) * (small_font.get_height() + 4)
    cy = content_rect.centery - total_h // 2
    for line in lines:
        s = small_font.render(line, True, DIM_TEXT)
        screen.blit(s, (content_rect.centerx - s.get_width() // 2, cy))
        cy += small_font.get_height() + 4
    return 1.0, 0, 0


def draw_status_bar(
    screen: pygame.Surface,
    fps: float,
    has_device: bool,
    rect: pygame.Rect,
    small_font: pygame.font.Font,
    hint: str = "[Q / ESC] quit",
    adb_fps: float = 0.0,
) -> None:
    """Draw the top status bar with app FPS and ADB capture FPS."""
    pygame.draw.rect(screen, STATUS_BG, rect)

    device_col = DEVICE_OK_COL if has_device else DEVICE_ERR_COL
    device_txt = "Device: connected" if has_device else "Device: not connected"
    d = small_font.render(device_txt, True, device_col)
    screen.blit(d, (rect.x + 12, rect.y + (rect.height - d.get_height()) // 2))

    # App FPS on the far right, ADB FPS just to its left
    app_fps_s = small_font.render(f"App {fps:.0f} fps", True, DIM_TEXT)
    adb_fps_col = DEVICE_OK_COL if adb_fps >= 5 else DEVICE_ERR_COL
    adb_fps_s = small_font.render(f"ADB {adb_fps:.1f} fps", True, adb_fps_col)

    app_x = rect.right - app_fps_s.get_width() - 12
    adb_x = app_x - adb_fps_s.get_width() - 16
    mid_y = rect.y + (rect.height - app_fps_s.get_height()) // 2
    screen.blit(app_fps_s, (app_x, mid_y))
    screen.blit(adb_fps_s, (adb_x, mid_y))

    hint_s = small_font.render(hint, True, DIM_TEXT)
    screen.blit(hint_s, (rect.centerx - hint_s.get_width() // 2,
                         rect.y + (rect.height - hint_s.get_height()) // 2))


def draw_grid_overlay(
    screen: pygame.Surface,
    box: CalibrationBox,
    scale: float,
    blit_x: int,
    blit_y: int,
) -> None:
    """Draw the calibrated 8×8 grid bounding box over the phone panel."""
    sr = box.to_screen_rect(scale, blit_x, blit_y)
    pygame.draw.rect(screen, OVERLAY_BOX, sr, width=2)

    cell_w = sr.width  / 8
    cell_h = sr.height / 8
    for i in range(1, 8):
        x = int(sr.x + i * cell_w)
        pygame.draw.line(screen, OVERLAY_GRID, (x, sr.y), (x, sr.bottom), 1)
        y = int(sr.y + i * cell_h)
        pygame.draw.line(screen, OVERLAY_GRID, (sr.x, y), (sr.right, y), 1)

    for row in range(8):
        for col in range(8):
            cx = int(sr.x + (col + 0.5) * cell_w)
            cy = int(sr.y + (row + 0.5) * cell_h)
            pygame.draw.circle(screen, OVERLAY_DOT, (cx, cy), 2)


def draw_queue_overlay(
    screen: pygame.Surface,
    box: CalibrationBox,
    scale: float,
    blit_x: int,
    blit_y: int,
    font: pygame.font.Font,
    chosen_slot: Optional[int] = None,
) -> None:
    """Draw the queue bounding box divided into 3 labeled piece slots."""
    sr     = box.to_screen_rect(scale, blit_x, blit_y)
    slot_w = sr.width / 3

    pygame.draw.rect(screen, QUEUE_BOX_COLOR, sr, width=2)
    for i in range(1, 3):
        x = int(sr.x + i * slot_w)
        pygame.draw.line(screen, QUEUE_DIVIDER, (x, sr.y), (x, sr.bottom), 1)

    if chosen_slot is not None and 0 <= chosen_slot < 3:
        slot_rect = pygame.Rect(int(sr.x + chosen_slot * slot_w), sr.y,
                                int(slot_w) + 1, sr.height)
        ov = pygame.Surface(slot_rect.size, pygame.SRCALPHA)
        ov.fill((*SUGGEST_FILL, 70))
        screen.blit(ov, slot_rect.topleft)
        pygame.draw.rect(screen, SUGGEST_BORDER, slot_rect, width=3)

    for i in range(3):
        col   = SUGGEST_FILL if i == chosen_slot else QUEUE_BOX_COLOR
        label = font.render(f"P{i + 1}", True, col)
        lx    = int(sr.x + (i + 0.5) * slot_w) - label.get_width() // 2
        ly    = max(0, sr.y - label.get_height() - 2)
        screen.blit(label, (lx, ly))


def draw_suggestion_on_phone(
    screen: pygame.Surface,
    grid_box: CalibrationBox,
    suggestion: Suggestion,
    scale: float,
    blit_x: int,
    blit_y: int,
) -> None:
    """Overlay the suggested placement as a ghost piece on the phone grid."""
    sr     = grid_box.to_screen_rect(scale, blit_x, blit_y)
    cell_w = sr.width  / 8
    cell_h = sr.height / 8
    for dr, dc in suggestion.piece.cells:
        r, c = suggestion.row + dr, suggestion.col + dc
        if not (0 <= r < 8 and 0 <= c < 8):
            continue
        rect = pygame.Rect(int(sr.x + c * cell_w) + 1, int(sr.y + r * cell_h) + 1,
                           int(cell_w) - 2, int(cell_h) - 2)
        ov = pygame.Surface(rect.size, pygame.SRCALPHA)
        ov.fill((*SUGGEST_FILL, SUGGEST_FILL_A))
        screen.blit(ov, rect.topleft)
        pygame.draw.rect(screen, SUGGEST_BORDER, rect, width=2)


_CALIB_FILL   = (255, 160, 0)
_CALIB_BORDER = (255, 220, 50)
_CALIB_FILL_A = 80


def draw_calib_target_on_phone(
    screen: pygame.Surface,
    grid_box: CalibrationBox,
    row: int,
    col: int,
    scale: float,
    blit_x: int,
    blit_y: int,
) -> None:
    """Highlight the calibration target cell on the phone panel.

    Draws a pulsing-orange filled rectangle so the human knows exactly which
    cell to drag the piece to.
    """
    sr     = grid_box.to_screen_rect(scale, blit_x, blit_y)
    cell_w = sr.width  / 8
    cell_h = sr.height / 8
    rect   = pygame.Rect(
        int(sr.x + col * cell_w) + 1,
        int(sr.y + row * cell_h) + 1,
        int(cell_w) - 2,
        int(cell_h) - 2,
    )
    ov = pygame.Surface(rect.size, pygame.SRCALPHA)
    ov.fill((*_CALIB_FILL, _CALIB_FILL_A))
    screen.blit(ov, rect.topleft)
    pygame.draw.rect(screen, _CALIB_BORDER, rect, width=3)


_SWIPE_HEAD = (90, 255, 220)
_SWIPE_TAIL = (40, 180, 255)
_SWIPE_TTL_MS = 1500   # how long the arrow lingers after the swipe starts


def draw_swipe_arrow_on_phone(
    screen: pygame.Surface,
    src_frame_xy: tuple[int, int],
    dst_frame_xy: tuple[int, int],
    age_ms: int,
    scale: float,
    blit_x: int,
    blit_y: int,
    duration_ms: int = 350,
    small_font: Optional[pygame.font.Font] = None,
) -> None:
    """Draw the auto-play swipe path on the phone panel.

    Coordinates are in *frame pixels* (device coordinate space).  The arrow
    fades out over :data:`_SWIPE_TTL_MS`.  An animated dot interpolates from
    src→dst during ``duration_ms`` to visualise the in-flight finger.
    """
    if age_ms < 0 or age_ms > _SWIPE_TTL_MS:
        return

    # Frame pixels → screen pixels
    sx = blit_x + src_frame_xy[0] * scale
    sy = blit_y + src_frame_xy[1] * scale
    dx = blit_x + dst_frame_xy[0] * scale
    dy = blit_y + dst_frame_xy[1] * scale

    # Alpha fades linearly to 0 over the TTL
    alpha = max(0, min(255, int(255 * (1.0 - age_ms / _SWIPE_TTL_MS))))
    if alpha == 0:
        return

    # Build a transparent overlay sized to the line bounding box (cheap).
    pad = 12
    lo_x = int(min(sx, dx)) - pad
    lo_y = int(min(sy, dy)) - pad
    hi_x = int(max(sx, dx)) + pad
    hi_y = int(max(sy, dy)) + pad
    w, h = max(1, hi_x - lo_x), max(1, hi_y - lo_y)
    ov   = pygame.Surface((w, h), pygame.SRCALPHA)

    p1 = (int(sx - lo_x), int(sy - lo_y))
    p2 = (int(dx - lo_x), int(dy - lo_y))

    # Thick line with a glow effect (two passes)
    pygame.draw.line(ov, (*_SWIPE_TAIL, alpha // 3), p1, p2, 9)
    pygame.draw.line(ov, (*_SWIPE_HEAD, alpha),     p1, p2, 3)

    # Start dot (origin)
    pygame.draw.circle(ov, (*_SWIPE_TAIL, alpha), p1, 7, width=2)
    # End dot (destination)
    pygame.draw.circle(ov, (*_SWIPE_HEAD, alpha), p2, 8)
    pygame.draw.circle(ov, (255, 255, 255, alpha), p2, 4)

    # Animated "finger" dot moving along the path during the swipe
    if 0 <= age_ms <= duration_ms and duration_ms > 0:
        t = age_ms / duration_ms
        fx = p1[0] + (p2[0] - p1[0]) * t
        fy = p1[1] + (p2[1] - p1[1]) * t
        pygame.draw.circle(ov, (255, 255, 255, alpha), (int(fx), int(fy)), 10)
        pygame.draw.circle(ov, (*_SWIPE_HEAD, alpha),  (int(fx), int(fy)), 10, width=2)

    screen.blit(ov, (lo_x, lo_y))

    # Optional coord label near the destination
    if small_font is not None:
        label = f"({dst_frame_xy[0]},{dst_frame_xy[1]})"
        text  = small_font.render(label, True, _SWIPE_HEAD)
        text.set_alpha(alpha)
        screen.blit(text, (int(dx) + 12, int(dy) + 4))


def draw_servo_error_on_phone(
    screen: pygame.Surface,
    *,
    target_xy: Optional[tuple[int, int]],
    measured_xy: Optional[tuple[int, int]],
    target_cells: Optional[list[tuple[int, int]]] = None,
    measured_cells: Optional[list[tuple[int, int]]] = None,
    scale: float,
    blit_x: int,
    blit_y: int,
    small_font: Optional[pygame.font.Font] = None,
) -> None:
    """Draw the servo error visualization on the phone panel.

    Per cell: a green crosshair at the target cell centre, a magenta
    dot at where the matcher's per-cell COM lands, and a yellow line
    between them.  Plus a thicker aggregate error line between the
    means of both sets so the bulk-motion direction is obvious at a
    glance.  Coordinates are in frame pixels.
    """
    target_col   = (60,  220, 130)   # bright green = where we want the piece
    measured_col = (255, 60,  150)   # hot magenta = where the piece is
    line_col     = (255, 230, 60)    # yellow = error vector

    def _to_screen(p: tuple[int, int]) -> tuple[int, int]:
        return (int(blit_x + p[0] * scale), int(blit_y + p[1] * scale))

    # ── Per-cell error lines (thin) ────────────────────────────────
    if target_cells and measured_cells and len(target_cells) == len(measured_cells):
        for t_xy, m_xy in zip(target_cells, measured_cells):
            t = _to_screen(t_xy)
            m = _to_screen(m_xy)
            pygame.draw.line(screen, line_col, m, t, 2)

    # ── Aggregate error line (thick, glow) ─────────────────────────
    if target_xy is not None and measured_xy is not None:
        t = _to_screen(target_xy)
        m = _to_screen(measured_xy)
        pygame.draw.line(screen, (line_col[0] // 2, line_col[1] // 2, 0), m, t, 7)
        pygame.draw.line(screen, line_col, m, t, 3)

    # ── Target dots (one per cell) ─────────────────────────────────
    if target_cells:
        for t_xy in target_cells:
            t = _to_screen(t_xy)
            pygame.draw.circle(screen, (0, 0, 0), t, 7)
            pygame.draw.circle(screen, target_col, t, 5)

    # Aggregate target — bigger crosshair so it stands out.
    if target_xy is not None:
        t = _to_screen(target_xy)
        pygame.draw.circle(screen, (0, 0, 0), t, 11)
        pygame.draw.circle(screen, target_col, t, 9)
        pygame.draw.line(screen, target_col, (t[0] - 14, t[1]), (t[0] + 14, t[1]), 2)
        pygame.draw.line(screen, target_col, (t[0], t[1] - 14), (t[0], t[1] + 14), 2)

    # ── Measured dots (one per cell) ───────────────────────────────
    if measured_cells:
        for m_xy in measured_cells:
            m = _to_screen(m_xy)
            pygame.draw.circle(screen, (0, 0, 0), m, 7)
            pygame.draw.circle(screen, measured_col, m, 5)

    if measured_xy is not None:
        m = _to_screen(measured_xy)
        pygame.draw.circle(screen, (0, 0, 0), m, 11)
        pygame.draw.circle(screen, measured_col, m, 8)
        pygame.draw.circle(screen, (255, 255, 255), m, 3)

    if small_font is not None and target_xy is not None and measured_xy is not None:
        err_x = target_xy[0] - measured_xy[0]
        err_y = target_xy[1] - measured_xy[1]
        cells_label = ""
        if measured_cells is not None and target_cells is not None:
            cells_label = f"  cells={len(measured_cells)}/{len(target_cells)}"
        label = f"err=({err_x:+d},{err_y:+d}){cells_label}"
        text  = small_font.render(label, True, line_col)
        bg    = pygame.Surface(
            (text.get_width() + 8, text.get_height() + 4), pygame.SRCALPHA,
        )
        bg.fill((0, 0, 0, 180))
        mid_x = (target_xy[0] + measured_xy[0]) / 2
        mid_y = (target_xy[1] + measured_xy[1]) / 2
        sx = int(blit_x + mid_x * scale)
        sy = int(blit_y + mid_y * scale)
        screen.blit(bg,   (sx + 6, sy + 6))
        screen.blit(text, (sx + 10, sy + 8))


def draw_drag_preview(
    screen: pygame.Surface,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """Draw a semi-transparent rectangle while the user is dragging."""
    x1, y1 = min(start[0], end[0]), min(start[1], end[1])
    x2, y2 = max(start[0], end[0]), max(start[1], end[1])
    w, h   = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return
    ov = pygame.Surface((w, h), pygame.SRCALPHA)
    ov.fill((255, 200, 0, 50))
    screen.blit(ov, (x1, y1))
    pygame.draw.rect(screen, DRAG_COLOR, (x1, y1, w, h), width=2)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _wrap(text: str, font: pygame.font.Font, max_w: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_w:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]
