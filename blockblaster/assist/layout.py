"""Window layout constants and make_window() for the assist GUI."""

from __future__ import annotations

import pygame

# Phone panel: portrait aspect close to iPhone (9:19.5)
PHONE_ASPECT_W = 9
PHONE_ASPECT_H = 19.5

PHONE_PANEL_H = 780          # target height for the phone panel
PHONE_PANEL_W = int(PHONE_PANEL_H * PHONE_ASPECT_W / PHONE_ASPECT_H)

# Right panel must fit: board(60×8=480) + gap(12) + queue(200) + inner padding(2×20=40) = 732
RECON_PANEL_W = 760          # right panel width (board + queue side-by-side)
PANEL_PAD     = 20           # padding around each panel
STATUS_BAR_H  = 36           # top strip: FPS + device indicator
CONTROLS_H    = 80           # clickable chip panel below the status bar

WIN_W = PANEL_PAD + PHONE_PANEL_W + PANEL_PAD + RECON_PANEL_W + PANEL_PAD
WIN_H = PANEL_PAD + PHONE_PANEL_H + PANEL_PAD + STATUS_BAR_H + CONTROLS_H

# Pre-built rects for each region
PHONE_RECT = pygame.Rect(PANEL_PAD, PANEL_PAD, PHONE_PANEL_W, PHONE_PANEL_H)
RECON_RECT = pygame.Rect(
    PANEL_PAD + PHONE_PANEL_W + PANEL_PAD,
    PANEL_PAD,
    RECON_PANEL_W,
    PHONE_PANEL_H,
)
STATUS_RECT   = pygame.Rect(0, PANEL_PAD + PHONE_PANEL_H + PANEL_PAD, WIN_W, STATUS_BAR_H)
CONTROLS_RECT = pygame.Rect(0, STATUS_RECT.bottom, WIN_W, CONTROLS_H)

BG_COLOR = (14, 14, 20)


def make_window() -> pygame.Surface:
    """Create and return the main pygame surface."""
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Block Blast – Assist")
    return screen
