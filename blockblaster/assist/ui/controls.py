"""Controls panel for the assist GUI."""

from __future__ import annotations

import pygame

_BG         = (22, 22, 32)
_PANEL_LINE = (50, 50, 70)
_WHITE      = (220, 220, 230)
_RED        = (210, 70, 70)
_GREEN      = (80, 220, 120)
_GOLD       = (255, 200, 40)
_ON_FILL    = (40, 70, 50)
_OFF_FILL   = (35, 35, 50)

_CHIP_H   = 36
_CHIP_PAD = 10
_CHIP_GAP = 8


def draw_controls_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    *,
    autoplay_on: bool = False,
    servo_busy: bool = False,
    show_debug: bool = False,
    edit_board: bool = False,
) -> dict[str, pygame.Rect]:
    """Draw clickable chip-buttons and return their screen rects.

    The Autoplay chip highlights green while the toggle is ON (● = a placement
    is running right now); the Debug chip highlights gold while the
    servo-tracking overlay is enabled.
    """
    pygame.draw.rect(screen, _BG, rect)
    pygame.draw.line(screen, _PANEL_LINE, rect.topleft, rect.topright)

    if autoplay_on:
        autoplay_label = "Autoplay [A] ON ●" if servo_busy else "Autoplay [A] ON"
    else:
        autoplay_label = "Autoplay [A] OFF"
    chips: list[tuple[str, str, tuple[int, int, int], bool]] = [
        ("autoplay",    autoplay_label,    _GREEN, autoplay_on),
        ("debug",       "Debug [D]",       _GOLD,  show_debug),
        ("editboard",   "Edit Board [E]",  _GOLD,  edit_board),
        ("recalibrate", "Recalibrate [R]", _WHITE, False),
        ("screenshot",  "Screenshot [S]",  _WHITE, False),
        ("quit",        "Quit  [Q]",       _RED,   False),
    ]

    cy = rect.y + rect.height // 2
    x  = rect.x + _CHIP_GAP

    result: dict[str, pygame.Rect] = {}
    for key, label, colour, active in chips:
        text_surf = font.render(label, True, colour)
        w = text_surf.get_width() + _CHIP_PAD * 2
        chip_rect = pygame.Rect(x, cy - _CHIP_H // 2, w, _CHIP_H)

        fill = _ON_FILL if active else _OFF_FILL
        pygame.draw.rect(screen, fill, chip_rect, border_radius=6)
        pygame.draw.rect(screen, colour, chip_rect,
                         width=2 if active else 1, border_radius=6)
        screen.blit(text_surf, (
            chip_rect.x + _CHIP_PAD,
            chip_rect.y + (_CHIP_H - text_surf.get_height()) // 2,
        ))

        result[key] = chip_rect
        x += w + _CHIP_GAP

    return result
