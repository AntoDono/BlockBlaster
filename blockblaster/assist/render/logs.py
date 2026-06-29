"""Scrolling log panel for the assist GUI."""

from __future__ import annotations

from collections import deque

import pygame

from blockblaster.assist.render.phone import DIM_TEXT, LABEL_COL, PANEL_BG, PANEL_BORDER

_LINE_GAP = 2
_SCROLL_LINES_PER_TICK = 3


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


def _wrapped_lines(lines: deque[str], font: pygame.font.Font, max_w: int) -> list[str]:
    out: list[str] = []
    for line in lines:
        if font.size(line)[0] <= max_w:
            out.append(line)
        else:
            out.extend(_wrap(line, font, max_w))
    return out


def draw_log_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    lines: deque[str],
    small_font: pygame.font.Font,
    scroll_from_bottom: int,
) -> int:
    """Draw the log; return clamped ``scroll_from_bottom`` (0 = newest at bottom)."""
    pygame.draw.rect(screen, PANEL_BG, rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    lbl = small_font.render("LOG", True, LABEL_COL)
    screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.y + 8))

    hint = small_font.render("scroll", True, (70, 70, 90))
    screen.blit(hint, (rect.right - hint.get_width() - 10, rect.y + 8))

    content = pygame.Rect(rect.x + 8, rect.y + 30, rect.width - 16, rect.height - 38)
    if not lines:
        msg = small_font.render("—", True, DIM_TEXT)
        screen.blit(msg, (content.x + 4, content.y + 4))
        return 0

    line_h = small_font.get_height() + _LINE_GAP
    wrapped = _wrapped_lines(lines, small_font, content.width - 4)
    max_visible = max(1, content.height // line_h)
    max_scroll = max(0, len(wrapped) - max_visible)
    scroll = max(0, min(scroll_from_bottom, max_scroll))
    start = len(wrapped) - max_visible - scroll
    visible = wrapped[max(0, start): max(0, start) + max_visible]

    prev_clip = screen.get_clip()
    screen.set_clip(content)
    y = content.y
    for line in visible:
        screen.blit(small_font.render(line, True, DIM_TEXT), (content.x + 2, y))
        y += line_h
    screen.set_clip(prev_clip)

    if max_scroll > 0:
        track = pygame.Rect(content.right - 5, content.y, 4, content.height)
        pygame.draw.rect(screen, (35, 35, 50), track, border_radius=2)
        thumb_h = max(12, int(content.height * max_visible / len(wrapped)))
        thumb_y = content.y + int(
            (content.height - thumb_h) * (max_scroll - scroll) / max_scroll
        ) if max_scroll else content.y
        pygame.draw.rect(screen, (90, 90, 115), (track.x, thumb_y, track.width, thumb_h),
                         border_radius=2)

    return scroll
