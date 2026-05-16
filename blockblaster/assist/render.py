"""Rendering helpers for the assist GUI panels."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pygame

from blockblaster.assist.calibration import CalibrationBox
from blockblaster.game.board import Board
from blockblaster.game.pieces import Piece
from blockblaster.gui.render import (
    BG_COLOR as GUI_BG,
    CELL_SIZE,
    GRID_PADDING,
    PIECE_COLORS,
    QUEUE_WIDTH,
    draw_board,
    draw_piece_preview,
)

# ── Colours ──────────────────────────────────────────────────────────────────
PANEL_BG        = (20, 20, 30)
PANEL_BORDER    = (55, 55, 75)
TEXT_COLOR      = (220, 220, 235)
DIM_TEXT        = (110, 110, 130)
STATUS_BG       = (10, 10, 16)
DEVICE_OK_COL   = (60, 220, 100)
DEVICE_ERR_COL  = (220, 70, 70)
LABEL_COL       = (160, 160, 185)
OVERLAY_BOX     = (0, 240, 220)      # cyan bounding box (grid)
OVERLAY_GRID    = (0, 180, 160)      # internal cell lines (grid)
OVERLAY_DOT     = (255, 255, 100)    # cell-centre dots (grid)
DRAG_COLOR      = (255, 200, 0)      # drag-in-progress rect
QUEUE_BOX_COLOR = (255, 210, 50)     # yellow bounding box (queue)
QUEUE_DIVIDER   = (200, 165, 40)     # vertical dividers between piece slots


# ── Frame → surface ───────────────────────────────────────────────────────────

def bgr_to_surface(
    frame_bgr: np.ndarray,
    target_rect: pygame.Rect,
) -> tuple[pygame.Surface, float, int, int]:
    """Convert a BGR frame to a pygame.Surface letterboxed into target_rect.

    Returns:
        (surface, scale, blit_x, blit_y)

    scale   – pixels-per-frame-pixel for the rendered image.
    blit_x  – screen x where the top-left of the surface is drawn.
    blit_y  – screen y where the top-left of the surface is drawn.
    """
    h, w = frame_bgr.shape[:2]
    scale = min(target_rect.width / w, target_rect.height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    surf = pygame.image.frombuffer(rgb.tobytes(), (new_w, new_h), "RGB")
    blit_x = target_rect.x + (target_rect.width - new_w) // 2
    blit_y = target_rect.y + (target_rect.height - new_h) // 2
    return surf, scale, blit_x, blit_y


def draw_phone_panel(
    screen: pygame.Surface,
    frame: Optional[np.ndarray],
    rect: pygame.Rect,
    error_msg: Optional[str],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> tuple[float, int, int]:
    """Draw the left phone-screen panel.

    Returns (scale, blit_x, blit_y) so callers can map mouse coordinates back
    to frame pixel coordinates.  When no frame is available, returns (1.0, 0, 0).
    """
    # Background + border
    pygame.draw.rect(screen, PANEL_BG, rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    # Label at top
    lbl = small_font.render("PHONE SCREEN", True, LABEL_COL)
    screen.blit(lbl, (rect.x + 10, rect.y + 8))

    content_rect = pygame.Rect(rect.x + 4, rect.y + 30, rect.width - 8, rect.height - 38)

    if frame is not None:
        surf, scale, blit_x, blit_y = bgr_to_surface(frame, content_rect)
        screen.blit(surf, (blit_x, blit_y))
        return scale, blit_x, blit_y
    else:
        # Placeholder text
        msg = error_msg if error_msg else "Waiting for device…"
        lines = _wrap(msg, small_font, content_rect.width - 20)
        total_h = len(lines) * (small_font.get_height() + 4)
        cy = content_rect.centery - total_h // 2
        for line in lines:
            surf_txt = small_font.render(line, True, DIM_TEXT)
            screen.blit(surf_txt, (content_rect.centerx - surf_txt.get_width() // 2, cy))
            cy += small_font.get_height() + 4
        return 1.0, 0, 0


def draw_recon_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    board: Board,
    queue: list[Piece],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    """Draw the right reconstructed game-state panel."""
    pygame.draw.rect(screen, PANEL_BG, rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    lbl = small_font.render("RECONSTRUCTED SCENE", True, LABEL_COL)
    screen.blit(lbl, (rect.x + 10, rect.y + 8))

    # Centre board + queue inside the panel
    board_px = CELL_SIZE * 8   # param.BOARD_SIZE = 8 (hardcoded to avoid import cycles)
    queue_panel_w = QUEUE_WIDTH
    total_inner_w = board_px + 12 + queue_panel_w
    board_x = rect.x + (rect.width - total_inner_w) // 2
    board_y = rect.y + 36 + (rect.height - 36 - board_px) // 2

    draw_board(screen, board, board_x, board_y)

    # Queue panel (minimal, re-implementing draw_queue inline at smaller scale)
    qx = board_x + board_px + 12
    _draw_mini_queue(screen, queue, qx, board_y, board_px, small_font)

    # Caption at the bottom
    caption = small_font.render("drag on phone panel to calibrate grid", True, DIM_TEXT)
    screen.blit(caption, (rect.centerx - caption.get_width() // 2, rect.bottom - 26))


def draw_status_bar(
    screen: pygame.Surface,
    fps: float,
    has_device: bool,
    rect: pygame.Rect,
    small_font: pygame.font.Font,
    hint: str = "[Q / ESC] quit",
) -> None:
    """Draw the bottom status bar."""
    pygame.draw.rect(screen, STATUS_BG, rect)

    device_col = DEVICE_OK_COL if has_device else DEVICE_ERR_COL
    device_txt = "Device: connected" if has_device else "Device: not connected"
    d_surf = small_font.render(device_txt, True, device_col)
    screen.blit(d_surf, (rect.x + 12, rect.y + (rect.height - d_surf.get_height()) // 2))

    fps_surf = small_font.render(f"{fps:.0f} FPS", True, DIM_TEXT)
    screen.blit(fps_surf, (rect.right - fps_surf.get_width() - 12,
                           rect.y + (rect.height - fps_surf.get_height()) // 2))

    hint_surf = small_font.render(hint, True, DIM_TEXT)
    screen.blit(hint_surf, (rect.centerx - hint_surf.get_width() // 2,
                            rect.y + (rect.height - hint_surf.get_height()) // 2))


# ── Calibration overlays ─────────────────────────────────────────────────────

def draw_grid_overlay(
    screen: pygame.Surface,
    box: CalibrationBox,
    scale: float,
    blit_x: int,
    blit_y: int,
) -> None:
    """Draw the calibrated bounding box + 8×8 cell grid over the phone panel."""
    sr = box.to_screen_rect(scale, blit_x, blit_y)

    # Outer bounding box
    pygame.draw.rect(screen, OVERLAY_BOX, sr, width=2)

    # Internal 7 vertical + 7 horizontal lines to show 8×8 cells
    cell_w = sr.width / 8
    cell_h = sr.height / 8
    for i in range(1, 8):
        # Vertical line
        x = int(sr.x + i * cell_w)
        pygame.draw.line(screen, OVERLAY_GRID, (x, sr.y), (x, sr.bottom), 1)
        # Horizontal line
        y = int(sr.y + i * cell_h)
        pygame.draw.line(screen, OVERLAY_GRID, (sr.x, y), (sr.right, y), 1)

    # Cell-centre dots
    for row in range(8):
        for col in range(8):
            cx = int(sr.x + (col + 0.5) * cell_w)
            cy = int(sr.y + (row + 0.5) * cell_h)
            pygame.draw.circle(screen, OVERLAY_DOT, (cx, cy), 2)


def draw_queue_overlay(
    screen: pygame.Surface,
    box: CalibrationBox,
    scale: float,
    blit_x: int,
    blit_y: int,
    font: pygame.font.Font,
) -> None:
    """Draw the queue bounding box divided into 3 labeled piece slots."""
    sr = box.to_screen_rect(scale, blit_x, blit_y)

    # Outer bounding box in yellow
    pygame.draw.rect(screen, QUEUE_BOX_COLOR, sr, width=2)

    # 2 vertical dividers splitting into 3 equal columns
    slot_w = sr.width / 3
    for i in range(1, 3):
        x = int(sr.x + i * slot_w)
        pygame.draw.line(screen, QUEUE_DIVIDER, (x, sr.y), (x, sr.bottom), 1)

    # Slot labels "P1", "P2", "P3" centred in each column
    for i in range(3):
        label_surf = font.render(f"P{i + 1}", True, QUEUE_BOX_COLOR)
        lx = int(sr.x + (i + 0.5) * slot_w) - label_surf.get_width() // 2
        ly = sr.y - label_surf.get_height() - 2
        # Keep label inside the screen
        ly = max(0, ly)
        screen.blit(label_surf, (lx, ly))


def draw_drag_preview(
    screen: pygame.Surface,
    start: tuple[int, int],
    end: tuple[int, int],
) -> None:
    """Draw a semi-transparent rectangle while the user is dragging."""
    x1, y1 = min(start[0], end[0]), min(start[1], end[1])
    x2, y2 = max(start[0], end[0]), max(start[1], end[1])
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return
    overlay = pygame.Surface((w, h), pygame.SRCALPHA)
    overlay.fill((255, 200, 0, 50))
    screen.blit(overlay, (x1, y1))
    pygame.draw.rect(screen, DRAG_COLOR, (x1, y1, w, h), width=2)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _wrap(text: str, font: pygame.font.Font, max_w: int) -> list[str]:
    """Naive word-wrap for status messages."""
    words = text.split()
    lines: list[str] = []
    current = ""
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


def _draw_mini_queue(
    surface: pygame.Surface,
    queue: list[Piece],
    x0: int,
    y0: int,
    panel_h: int,
    small_font: pygame.font.Font,
) -> None:
    """Minimal queue preview in the recon panel."""
    from blockblaster.gui.render import QUEUE_BG, QUEUE_BORDER, DIM_TEXT as GUI_DIM

    qw = QUEUE_WIDTH - 10
    pygame.draw.rect(surface, QUEUE_BG, (x0, y0, qw, panel_h), border_radius=8)
    pygame.draw.rect(surface, QUEUE_BORDER, (x0, y0, qw, panel_h), width=2, border_radius=8)

    lbl = small_font.render("NEXT", True, GUI_DIM)
    surface.blit(lbl, (x0 + 10, y0 + 8))

    if not queue:
        empty = small_font.render("—", True, GUI_DIM)
        surface.blit(empty, (x0 + qw // 2 - empty.get_width() // 2, y0 + panel_h // 2))
        return

    slot_h = (panel_h - 30) // max(len(queue), 1)
    for i, piece in enumerate(queue):
        sy = y0 + 30 + i * slot_h
        num = small_font.render(f"#{i + 1}", True, GUI_DIM)
        surface.blit(num, (x0 + 8, sy))
        draw_piece_preview(
            surface, piece,
            x0 + 10, sy + 18,
            color=PIECE_COLORS[i % len(PIECE_COLORS)],
        )
