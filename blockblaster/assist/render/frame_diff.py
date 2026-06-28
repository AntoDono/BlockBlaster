"""Frame-difference panel for the assist GUI.

Draws the motion highlight produced by
:class:`blockblaster.assist.vision.frame_diff.FrameDiffTracker`: moving pixels
(and their recently-cached positions) glow over a darkened copy of the frame.
"""

from __future__ import annotations

from typing import Optional

import pygame

from blockblaster.assist.render.phone import (
    DIM_TEXT,
    LABEL_COL,
    PANEL_BG,
    PANEL_BORDER,
    bgr_to_surface,
)
from blockblaster.assist.vision.frame_diff import (
    FrameDiffTracker,
    suggestion_cell_boxes,
)

# Suggestion outline colour — gold, kept distinct from the green recon ghost.
SUGGEST_OUTLINE = (255, 200, 40)
SUGGEST_OUTLINE_W = 3


def draw_frame_diff_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    tracker: FrameDiffTracker,
    now: float,
    small_font: pygame.font.Font,
    servo_debug: Optional[object] = None,
) -> None:
    """Draw the frame-difference panel.

    On top of the motion composite, the advisor's suggested placement is
    outlined directly over the phone screen (mapped from board pixels). The
    placement is read from the tracker, which holds the last one until it
    changes, so the outline stays put even while the analyzer briefly drops it.
    """
    pygame.draw.rect(screen, PANEL_BG,     rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    lbl = small_font.render("FRAME DIFF", True, LABEL_COL)
    screen.blit(lbl, (rect.x + 10, rect.y + 8))

    motion_pct = tracker.motion_fraction * 100.0
    pct = small_font.render(f"{motion_pct:4.1f}%", True, DIM_TEXT)
    screen.blit(pct, (rect.right - pct.get_width() - 10, rect.y + 8))

    content = pygame.Rect(rect.x + 4, rect.y + 30, rect.width - 8, rect.height - 38)

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

    if servo_debug is not None:
        _draw_servo_debug(screen, servo_debug, scale, bx, by, small_font)

    if tracker.event_active(now):
        _draw_event_banner(screen, content, small_font)


# Servo debug overlay colours.
_DBG_TARGET   = (255, 200, 40)    # gold — where each cell should land
_DBG_MEASURED = (60, 220, 255)    # cyan — measured motion centroid per cell
_DBG_FINGER   = (255, 80, 200)    # magenta — commanded finger position


def _draw_servo_debug(
    screen: pygame.Surface,
    dbg: object,
    scale: float,
    blit_x: int,
    blit_y: int,
    font: pygame.font.Font,
) -> None:
    """Visualize what the servo tracks: target cells, motion centroids, finger.

    ``dbg`` is a ``control.servo.ServoDebug`` (typed as object to avoid a
    control-layer import in the render module).
    """
    def to_screen(p) -> tuple[int, int]:
        return (blit_x + int(p[0] * scale), blit_y + int(p[1] * scale))

    def bbox_rect(b) -> pygame.Rect:
        x0, y0, x1, y1 = b
        sx0, sy0 = to_screen((x0, y0))
        sx1, sy1 = to_screen((x1, y1))
        return pygame.Rect(sx0, sy0, max(1, sx1 - sx0), max(1, sy1 - sy0))

    observe_bbox = getattr(dbg, "observe_bbox", None)
    board_aware = getattr(dbg, "board_aware", False)
    unobserved = getattr(dbg, "unobserved_cells", None) or []
    target_bbox = getattr(dbg, "target_bbox", None)
    measured_bbox = getattr(dbg, "measured_bbox", None)
    finger = getattr(dbg, "finger_px", None)

    # Observed region: only visualize the focused local window in board-aware
    # mode — dim everything outside it plus the unobserved (filled) cells, so
    # only the searched area stays bright. No overlay during full-board scan.
    if observe_bbox is not None and board_aware:
        shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        full = screen.get_rect()
        obs = bbox_rect(observe_bbox)
        for r in (
            pygame.Rect(full.left, full.top, full.width, obs.top - full.top),
            pygame.Rect(full.left, obs.bottom, full.width, full.bottom - obs.bottom),
            pygame.Rect(full.left, obs.top, obs.left - full.left, obs.height),
            pygame.Rect(obs.right, obs.top, full.right - obs.right, obs.height),
        ):
            if r.width > 0 and r.height > 0:
                shade.fill((0, 0, 0, 110), r)
        for cell in unobserved:
            shade.fill((0, 0, 0, 150), bbox_rect(cell))
        screen.blit(shade, (0, 0))
        pygame.draw.rect(screen, (80, 240, 120), obs, width=2)
    score = getattr(dbg, "score", 0.0)
    locked = getattr(dbg, "locked", False)
    err = getattr(dbg, "err_px", (0, 0))
    step = getattr(dbg, "step_px", (0, 0))

    target_pts = getattr(dbg, "target_pts", None) or []
    measured_pts = getattr(dbg, "measured_pts", None) or []

    # Target footprint: hollow gold rectangle + its 5 reference points.
    if target_bbox is not None:
        pygame.draw.rect(screen, _DBG_TARGET, bbox_rect(target_bbox), width=2)
    for p in target_pts:
        pygame.draw.circle(screen, _DBG_TARGET, to_screen(p), 4, width=1)

    # Measured piece extent: cyan rectangle + its 5 reference points, each tied
    # to the corresponding target point by a line (the 5-point error mapping).
    if measured_bbox is not None:
        pygame.draw.rect(screen, _DBG_MEASURED, bbox_rect(measured_bbox), width=2)
    for i, p in enumerate(measured_pts):
        sp = to_screen(p)
        pygame.draw.circle(screen, _DBG_MEASURED, sp, 3)
        if i < len(target_pts):
            pygame.draw.line(screen, _DBG_MEASURED, sp, to_screen(target_pts[i]), 1)

    if finger is not None:
        fp = to_screen(finger)
        pygame.draw.line(screen, _DBG_FINGER, (fp[0] - 8, fp[1]), (fp[0] + 8, fp[1]), 2)
        pygame.draw.line(screen, _DBG_FINGER, (fp[0], fp[1] - 8), (fp[0], fp[1] + 8), 2)

    col = (80, 240, 120) if locked else _DBG_MEASURED
    mode = "BOARD-AWARE" if board_aware else "FULL SCAN"
    state = "LOCK" if locked else "track"
    dist = int(round((err[0] ** 2 + err[1] ** 2) ** 0.5))
    line1 = f"servo {state}  [{mode}]  s={score:.2f}"
    line2 = f"err=({err[0]:+d},{err[1]:+d})  corr=({step[0]:+d},{step[1]:+d})"
    line3 = f"dist to target: {dist}px"
    lh = font.get_height() + 2
    screen.blit(font.render(line1, True, col), (blit_x + 6, blit_y + 6))
    screen.blit(font.render(line2, True, col), (blit_x + 6, blit_y + 6 + lh))
    screen.blit(font.render(line3, True, col), (blit_x + 6, blit_y + 6 + 2 * lh))


def _draw_suggestion_outline(
    screen: pygame.Surface,
    suggestion: Optional[object],
    board_bbox: Optional[tuple[int, int, int, int]],
    scale: float,
    blit_x: int,
    blit_y: int,
) -> None:
    """Trace a gold outline around the suggested placement footprint."""
    boxes = suggestion_cell_boxes(suggestion, board_bbox)
    if not boxes:
        return

    # Map each in-bounds piece cell (by board grid coords) to its screen rect.
    occupied: dict[tuple[int, int], pygame.Rect] = {}
    for (dr, dc), (x, y, w, h) in zip(suggestion.piece.cells, boxes):
        rect = pygame.Rect(
            blit_x + int(x * scale),
            blit_y + int(y * scale),
            max(1, int(w * scale)),
            max(1, int(h * scale)),
        )
        occupied[(suggestion.row + dr, suggestion.col + dc)] = rect

    # Draw only the perimeter: skip an edge when the adjacent grid cell is also
    # part of the footprint, so internal grid lines stay hidden.
    for (r, c), cell in occupied.items():
        if (r, c - 1) not in occupied:
            pygame.draw.line(screen, SUGGEST_OUTLINE, cell.topleft, cell.bottomleft, SUGGEST_OUTLINE_W)
        if (r, c + 1) not in occupied:
            pygame.draw.line(screen, SUGGEST_OUTLINE, cell.topright, cell.bottomright, SUGGEST_OUTLINE_W)
        if (r - 1, c) not in occupied:
            pygame.draw.line(screen, SUGGEST_OUTLINE, cell.topleft, cell.topright, SUGGEST_OUTLINE_W)
        if (r + 1, c) not in occupied:
            pygame.draw.line(screen, SUGGEST_OUTLINE, cell.bottomleft, cell.bottomright, SUGGEST_OUTLINE_W)


_EVENT_BG     = (200, 40, 40)
_EVENT_TEXT   = (255, 240, 200)


def _draw_event_banner(
    screen: pygame.Surface,
    content: pygame.Rect,
    small_font: pygame.font.Font,
) -> None:
    label = small_font.render("EVENT DETECTED", True, _EVENT_TEXT)
    pad   = 8
    box   = pygame.Rect(0, 0, label.get_width() + pad * 2, label.get_height() + pad)
    box.centerx = content.centerx
    box.y = content.y + 12
    pygame.draw.rect(screen, _EVENT_BG, box, border_radius=6)
    screen.blit(label, (box.x + pad, box.y + pad // 2))
