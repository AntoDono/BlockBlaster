"""Controls panel for the assist GUI."""

from __future__ import annotations

import pygame

from blockblaster.assist.app_state import MODE_GRID


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_BG         = (22, 22, 32)
_PANEL_LINE = (50, 50, 70)
_WHITE      = (220, 220, 230)

_GREEN   = (60, 210, 100)
_RED     = (210, 70, 70)
_GREY    = (80, 80, 100)

_CHIP_H   = 36
_CHIP_PAD = 10
_CHIP_GAP = 8


# ---------------------------------------------------------------------------
# Controls panel (clickable chips)
# ---------------------------------------------------------------------------

def draw_controls_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    *,
    calib_mode: str,
    auto_enabled: bool,
    device_supports_input: bool,
    servo_debug_view: bool = False,
) -> dict[str, pygame.Rect]:
    """Draw the row of clickable chip-buttons and return their screen rects.

    Returns a dict mapping action name → ``pygame.Rect``.  Actions:
        ``"auto"``, ``"mode"``, ``"clear"``, ``"debug"``,
        ``"screenshot"``, ``"quit"``
    """
    pygame.draw.rect(screen, _BG, rect)
    pygame.draw.line(screen, _PANEL_LINE, rect.topleft, rect.topright)

    chips: list[tuple[str, str, tuple[int, int, int], bool]] = []

    if device_supports_input:
        if auto_enabled:
            chips.append(("auto", "Auto-play [ON]",  _GREEN, True))
        else:
            chips.append(("auto", "Auto-play [OFF]", _RED,   True))
    else:
        chips.append(("auto", "Auto-play [n/a]", _GREY, False))

    chips.append(("mode",  f"Mode: {'GRID' if calib_mode == MODE_GRID else 'PIECES'}", _WHITE, True))
    chips.append(("clear", "Clear box",  _WHITE, True))
    chips.append(("debug", "Dump debug", _WHITE, True))
    if servo_debug_view:
        chips.append(("servo_dbg", "Servo dbg [ON]", _GREEN, True))
    else:
        chips.append(("servo_dbg", "Servo dbg [V]", _WHITE, True))
    chips.append(("screenshot", "Screenshot [S]", _WHITE, True))
    chips.append(("quit", "Quit  [Q]", _RED, True))

    cy = rect.y + rect.height // 2
    x  = rect.x + _CHIP_GAP

    result: dict[str, pygame.Rect] = {}
    for key, label, colour, enabled in chips:
        text_surf = font.render(label, True, colour if enabled else _GREY)
        w = text_surf.get_width() + _CHIP_PAD * 2
        h = _CHIP_H
        chip_rect = pygame.Rect(x, cy - h // 2, w, h)

        bg_col = (35, 35, 50) if enabled else (25, 25, 35)
        pygame.draw.rect(screen, bg_col, chip_rect, border_radius=6)
        border_col = colour if enabled else _GREY
        pygame.draw.rect(screen, border_col, chip_rect, width=1, border_radius=6)

        screen.blit(text_surf, (
            chip_rect.x + _CHIP_PAD,
            chip_rect.y + (h - text_surf.get_height()) // 2,
        ))

        result[key] = chip_rect
        x += w + _CHIP_GAP

    return result
