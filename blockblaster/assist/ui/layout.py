"""Window layout constants and make_window() for the assist GUI."""

from __future__ import annotations

import pygame

# Phone panel: portrait aspect close to iPhone (9:19.5)
PHONE_ASPECT_W = 9
PHONE_ASPECT_H = 19.5

PHONE_PANEL_H = 780
PHONE_PANEL_W = int(PHONE_PANEL_H * PHONE_ASPECT_W / PHONE_ASPECT_H)

RECON_PANEL_W      = PHONE_PANEL_W
FRAME_DIFF_PANEL_W = PHONE_PANEL_W
SIDE_PANEL_W       = int(PHONE_PANEL_W * 0.82)  # CNN + log column (was 0.62)
PANEL_PAD           = 20
STATUS_BAR_H        = 36
CONTROLS_H          = 80
SIDE_COL_GAP        = 8

WIN_W = (
    PANEL_PAD + PHONE_PANEL_W + PANEL_PAD
    + RECON_PANEL_W + PANEL_PAD
    + FRAME_DIFF_PANEL_W + PANEL_PAD
    + SIDE_PANEL_W + PANEL_PAD
)
WIN_H = PANEL_PAD + PHONE_PANEL_H + PANEL_PAD + STATUS_BAR_H + CONTROLS_H

PHONE_RECT = pygame.Rect(PANEL_PAD, PANEL_PAD, PHONE_PANEL_W, PHONE_PANEL_H)
RECON_RECT = pygame.Rect(
    PHONE_RECT.right + PANEL_PAD,
    PANEL_PAD,
    RECON_PANEL_W,
    PHONE_PANEL_H,
)
FRAME_DIFF_RECT = pygame.Rect(
    RECON_RECT.right + PANEL_PAD,
    PANEL_PAD,
    FRAME_DIFF_PANEL_W,
    PHONE_PANEL_H,
)
_side_x = FRAME_DIFF_RECT.right + PANEL_PAD
_side_half_h = (PHONE_PANEL_H - SIDE_COL_GAP) // 2
CNN_DEBUG_RECT = pygame.Rect(
    _side_x,
    PANEL_PAD,
    SIDE_PANEL_W,
    _side_half_h,
)
LOG_RECT = pygame.Rect(
    _side_x,
    PANEL_PAD + _side_half_h + SIDE_COL_GAP,
    SIDE_PANEL_W,
    PHONE_PANEL_H - _side_half_h - SIDE_COL_GAP,
)
STATUS_RECT   = pygame.Rect(0, PANEL_PAD + PHONE_PANEL_H + PANEL_PAD, WIN_W, STATUS_BAR_H)
CONTROLS_RECT = pygame.Rect(0, STATUS_RECT.bottom, WIN_W, CONTROLS_H)

BG_COLOR = (14, 14, 20)


def make_window(fullscreen: bool = True) -> pygame.Surface:
    """Create and return the main pygame surface.

    Uses the SCALED flag so the fixed ``WIN_W × WIN_H`` logical layout is scaled
    to fill the display (letterboxed to preserve aspect). SCALED also maps mouse
    events back to logical coordinates, so the board-edit drag math is unchanged.
    """
    flags = pygame.SCALED
    if fullscreen:
        flags |= pygame.FULLSCREEN
    screen = pygame.display.set_mode((WIN_W, WIN_H), flags)
    pygame.display.set_caption("Block Blast – Assist")
    return screen
