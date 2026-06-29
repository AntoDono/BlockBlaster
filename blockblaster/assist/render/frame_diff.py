"""Frame-difference panel for the assist GUI.

Darkens the latest frame and glows the moving pixels (cached up to
``FrameDiffTracker.ttl`` seconds). The advisor's suggested placement is
outlined in gold on top, and the live servo state is drawn when active.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pygame

from blockblaster.assist.render.phone import (
    DIM_TEXT,
    LABEL_COL,
    PANEL_BG,
    PANEL_BORDER,
    SUGGEST_GOLD,
    bgr_to_surface,
    panel_content_rect,
)
from blockblaster.assist.vision.frame_diff import (
    FrameDiffTracker,
    suggestion_cell_boxes,
)

if TYPE_CHECKING:
    from blockblaster.assist.advisor import Suggestion
    from blockblaster.control.servo import ServoDebug

Bbox = tuple[int, int, int, int]  # (x, y, w, h) in frame pixels

SUGGEST_OUTLINE = SUGGEST_GOLD
SUGGEST_OUTLINE_W = 3

# Servo debug overlay colours.
_DBG_TARGET   = SUGGEST_GOLD      # gold — target cells
_DBG_MEASURED = (60, 220, 255)    # cyan — measured piece extent
_DBG_FINGER   = (255, 80, 200)    # magenta — commanded finger position

_EVENT_BG     = (200, 40, 40)
_EVENT_TEXT   = (255, 240, 200)

_STATUS_COLORS = {
    "BOUNDARY HIT": ((200, 50, 50), (255, 235, 220)),
    "FOCUSED":      ((150, 110, 20), (255, 240, 200)),
    "LOCKED":       ((30, 130, 70), (220, 255, 230)),
    "SEARCHING":    ((60, 60, 80), (200, 200, 220)),
    "TRAVELING":    ((30, 70, 110), (210, 235, 255)),
    "AREA GROW":    ((30, 130, 70), (220, 255, 230)),
    "ROW CLEAR":    ((120, 90, 30), (255, 245, 210)),
    "OFF BOARD":    ((150, 70, 20), (255, 230, 200)),
}


def draw_frame_diff_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    tracker: FrameDiffTracker,
    now: float,
    small_font: pygame.font.Font,
    servo_debug: Optional[ServoDebug] = None,
    servo_overlay_full: bool = False,
) -> None:
    """Draw the frame-difference panel.

    The suggestion outline is rendered here (not baked into the composite) so
    it stays sharp at panel scale. The placement is held by the tracker so it
    survives brief analyzer drops.
    """
    pygame.draw.rect(screen, PANEL_BG,     rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    lbl = small_font.render("FRAME DIFF", True, LABEL_COL)
    screen.blit(lbl, (rect.x + 10, rect.y + 8))

    motion_pct = tracker.motion_fraction * 100.0
    pct = small_font.render(f"{motion_pct:4.1f}%", True, DIM_TEXT)
    screen.blit(pct, (rect.right - pct.get_width() - 10, rect.y + 8))

    if servo_debug is not None:
        sc_col = (80, 240, 120) if servo_debug.locked else _DBG_MEASURED
        hdr = f"s={servo_debug.score:.2f}"
        if (
            servo_debug.initial_area_px > 0
            and servo_debug.current_area_px > 0
        ):
            ratio = servo_debug.current_area_px / servo_debug.initial_area_px
            hdr += f" a={ratio:.2f}×"
        elif servo_debug.initial_area_px > 0:
            hdr += f" a₀={servo_debug.initial_area_px}"
        score_lbl = small_font.render(hdr, True, sc_col)
        screen.blit(score_lbl, (rect.x + 10 + lbl.get_width() + 10, rect.y + 8))

    content = panel_content_rect(rect)

    composed = tracker.compose(now)
    if composed is None:
        s = small_font.render("Waiting for frames…", True, DIM_TEXT)
        screen.blit(s, (content.centerx - s.get_width() // 2,
                        content.centery - s.get_height() // 2))
        return

    surf, scale, bx, by = bgr_to_surface(composed, content)
    screen.blit(surf, (bx, by))

    _draw_suggestion_outline(
        screen, tracker.suggestion, tracker.board_bbox, scale, bx, by,
    )

    if servo_debug is not None and servo_overlay_full:
        _draw_servo_debug(screen, servo_debug, content, scale, bx, by, small_font)
    elif servo_debug is not None and servo_debug.status:
        _draw_status_banner(
            screen, servo_debug.status,
            content.x + 6, content.bottom - small_font.get_height() - 14,
            small_font,
        )

    if tracker.event_active(now):
        _draw_event_banner(screen, content, small_font)


def _draw_servo_debug(
    screen: pygame.Surface,
    dbg: ServoDebug,
    content: pygame.Rect,
    scale: float,
    blit_x: int,
    blit_y: int,
    font: pygame.font.Font,
) -> None:
    """Visualize what the servo tracks: target cells, motion blob, finger."""

    def to_screen(p: tuple[int, int]) -> tuple[int, int]:
        return (blit_x + int(p[0] * scale), blit_y + int(p[1] * scale))

    def bbox_rect(b: Bbox) -> pygame.Rect:
        x, y, w, h = b
        sx, sy = to_screen((x, y))
        sx1, sy1 = to_screen((x + w, y + h))
        return pygame.Rect(sx, sy, max(1, sx1 - sx), max(1, sy1 - sy))

    # In board-aware mode, dim everything outside the focus window and the
    # filled (unobserved) cells so only the searched area stays bright.
    if dbg.observe_bbox is not None and dbg.board_aware:
        shade = pygame.Surface((content.width, content.height), pygame.SRCALPHA)
        obs = bbox_rect(dbg.observe_bbox).clip(content)
        obs_local = obs.move(-content.x, -content.y)
        for r in (
            pygame.Rect(0, 0, content.width, obs_local.top),
            pygame.Rect(0, obs_local.bottom, content.width, content.height - obs_local.bottom),
            pygame.Rect(0, obs_local.top, obs_local.left, obs_local.height),
            pygame.Rect(obs_local.right, obs_local.top,
                        content.width - obs_local.right, obs_local.height),
        ):
            if r.width > 0 and r.height > 0:
                shade.fill((0, 0, 0, 110), r)
        for cell in dbg.unobserved_cells:
            cell_local = bbox_rect(cell).clip(content).move(-content.x, -content.y)
            shade.fill((0, 0, 0, 150), cell_local)
        screen.blit(shade, content.topleft)
        pygame.draw.rect(screen, (80, 240, 120), obs, width=2)

    if dbg.target_bbox is not None:
        pygame.draw.rect(screen, _DBG_TARGET, bbox_rect(dbg.target_bbox), width=2)
    for p in dbg.target_pts:
        pygame.draw.circle(screen, _DBG_TARGET, to_screen(p), 4, width=1)

    if dbg.measured_bbox is not None:
        pygame.draw.rect(screen, _DBG_MEASURED, bbox_rect(dbg.measured_bbox), width=2)
    for i, p in enumerate(dbg.measured_pts):
        sp = to_screen(p)
        pygame.draw.circle(screen, _DBG_MEASURED, sp, 3)
        if i < len(dbg.target_pts):
            pygame.draw.line(screen, _DBG_MEASURED, sp, to_screen(dbg.target_pts[i]), 1)

    if dbg.finger_px is not None:
        fp = to_screen(dbg.finger_px)
        pygame.draw.line(screen, _DBG_FINGER, (fp[0] - 8, fp[1]), (fp[0] + 8, fp[1]), 2)
        pygame.draw.line(screen, _DBG_FINGER, (fp[0], fp[1] - 8), (fp[0], fp[1] + 8), 2)

    col = (80, 240, 120) if dbg.locked else _DBG_MEASURED
    mode = "BOARD-AWARE" if dbg.board_aware else "FULL SCAN"
    state = "LOCK" if dbg.locked else "track"
    dist = int(round((dbg.err_px[0] ** 2 + dbg.err_px[1] ** 2) ** 0.5))
    lines = [
        f"servo {state}  [{mode}]  s={dbg.score:.2f}",
        f"err=({dbg.err_px[0]:+d},{dbg.err_px[1]:+d})  "
        f"corr=({dbg.step_px[0]:+d},{dbg.step_px[1]:+d})",
        f"dist to target: {dist}px",
    ]
    lh = font.get_height() + 2
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, col), (blit_x + 6, blit_y + 6 + i * lh))

    if dbg.status:
        _draw_status_banner(
            screen, dbg.status, blit_x, blit_y + 6 + len(lines) * lh + 4, font,
        )


def _draw_status_banner(
    screen: pygame.Surface, status: str, x: int, y: int,
    font: pygame.font.Font,
) -> None:
    bg, fg = (30, 70, 110), (210, 235, 255)
    for key, colors in _STATUS_COLORS.items():
        if status.startswith(key):
            bg, fg = colors
            break
    label = font.render(status, True, fg)
    box = pygame.Rect(x, y, label.get_width() + 14, label.get_height() + 8)
    pygame.draw.rect(screen, bg, box, border_radius=5)
    pygame.draw.rect(screen, fg, box, width=1, border_radius=5)
    screen.blit(label, (box.x + 7, box.y + 4))


def _draw_suggestion_outline(
    screen: pygame.Surface,
    suggestion: Optional[Suggestion],
    board_bbox: Optional[Bbox],
    scale: float,
    blit_x: int,
    blit_y: int,
) -> None:
    """Trace a gold outline around the suggested placement footprint."""
    boxes = suggestion_cell_boxes(suggestion, board_bbox)
    if not boxes or suggestion is None:
        return

    occupied: dict[tuple[int, int], pygame.Rect] = {}
    for (dr, dc), (x, y, w, h) in zip(suggestion.piece.cells, boxes):
        rect = pygame.Rect(
            blit_x + int(x * scale),
            blit_y + int(y * scale),
            max(1, int(w * scale)),
            max(1, int(h * scale)),
        )
        occupied[(suggestion.row + dr, suggestion.col + dc)] = rect

    # Draw an edge only when the adjacent cell is outside the footprint, so
    # internal seams stay hidden.
    for (r, c), cell in occupied.items():
        if (r, c - 1) not in occupied:
            pygame.draw.line(screen, SUGGEST_OUTLINE, cell.topleft, cell.bottomleft, SUGGEST_OUTLINE_W)
        if (r, c + 1) not in occupied:
            pygame.draw.line(screen, SUGGEST_OUTLINE, cell.topright, cell.bottomright, SUGGEST_OUTLINE_W)
        if (r - 1, c) not in occupied:
            pygame.draw.line(screen, SUGGEST_OUTLINE, cell.topleft, cell.topright, SUGGEST_OUTLINE_W)
        if (r + 1, c) not in occupied:
            pygame.draw.line(screen, SUGGEST_OUTLINE, cell.bottomleft, cell.bottomright, SUGGEST_OUTLINE_W)


def _draw_event_banner(
    screen: pygame.Surface,
    content: pygame.Rect,
    small_font: pygame.font.Font,
) -> None:
    label = small_font.render("EVENT DETECTED", True, _EVENT_TEXT)
    pad = 8
    box = pygame.Rect(0, 0, label.get_width() + pad * 2, label.get_height() + pad)
    box.centerx = content.centerx
    box.y = content.y + 12
    pygame.draw.rect(screen, _EVENT_BG, box, border_radius=6)
    screen.blit(label, (box.x + pad, box.y + pad // 2))
