"""Controls panel for the assist GUI."""

from __future__ import annotations

import pygame

_BG         = (22, 22, 32)
_PANEL_LINE = (50, 50, 70)
_WHITE      = (220, 220, 230)
_RED        = (210, 70, 70)

_CHIP_H   = 36
_CHIP_PAD = 10
_CHIP_GAP = 8


def draw_controls_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
) -> dict[str, pygame.Rect]:
    """Draw clickable chip-buttons and return their screen rects."""
    pygame.draw.rect(screen, _BG, rect)
    pygame.draw.line(screen, _PANEL_LINE, rect.topleft, rect.topright)

    chips: list[tuple[str, str, tuple[int, int, int]]] = [
        ("screenshot", "Screenshot [S]", _WHITE),
        ("quit",       "Quit  [Q]",      _RED),
    ]

    cy = rect.y + rect.height // 2
    x  = rect.x + _CHIP_GAP

    result: dict[str, pygame.Rect] = {}
    for key, label, colour in chips:
        text_surf = font.render(label, True, colour)
        w = text_surf.get_width() + _CHIP_PAD * 2
        chip_rect = pygame.Rect(x, cy - _CHIP_H // 2, w, _CHIP_H)

        pygame.draw.rect(screen, (35, 35, 50), chip_rect, border_radius=6)
        pygame.draw.rect(screen, colour,       chip_rect, width=1, border_radius=6)
        screen.blit(text_surf, (
            chip_rect.x + _CHIP_PAD,
            chip_rect.y + (_CHIP_H - text_surf.get_height()) // 2,
        ))

        result[key] = chip_rect
        x += w + _CHIP_GAP

    return result
