"""Frame-difference panel for the assist GUI.

Draws the motion highlight produced by
:class:`blockblaster.assist.vision.frame_diff.FrameDiffTracker`: moving pixels
(and their recently-cached positions) glow over a darkened copy of the frame.
"""

from __future__ import annotations

import pygame

from blockblaster.assist.render.phone import (
    DIM_TEXT,
    LABEL_COL,
    PANEL_BG,
    PANEL_BORDER,
    bgr_to_surface,
)
from blockblaster.assist.vision.frame_diff import FrameDiffTracker


def draw_frame_diff_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    tracker: FrameDiffTracker,
    now: float,
    small_font: pygame.font.Font,
) -> None:
    """Draw the frame-difference panel."""
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

    surf, _, bx, by = bgr_to_surface(composed, content)
    screen.blit(surf, (bx, by))

    if tracker.event_active(now):
        _draw_event_banner(screen, content, small_font)


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
