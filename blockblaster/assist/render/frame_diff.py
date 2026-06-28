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

    if tracker.event_active(now):
        _draw_event_banner(screen, content, small_font)


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
