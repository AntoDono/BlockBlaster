"""Pygame rendering helpers for the Block Blast board and queue."""

from __future__ import annotations

import pygame

import param
from blockblaster.game.board import Board
from blockblaster.game.pieces import Piece

# ── Palette ─────────────────────────────────────────────────────────────────
BG_COLOR       = (18,  18,  24)
GRID_BG        = (30,  30,  40)
GRID_LINE      = (50,  50,  65)
CELL_COLOR     = (80, 160, 240)
CELL_HIGHLIGHT = (120, 210, 255)
CELL_BORDER    = (40,  100, 180)
QUEUE_BG       = (24,  24,  34)
QUEUE_BORDER   = (60,  60,  80)
TEXT_COLOR     = (220, 220, 235)
DIM_TEXT       = (120, 120, 140)
GAME_OVER_COL  = (220,  60,  60)
PIECE_COLORS   = [
    (240, 100,  80),  # slot 0 – red-orange
    ( 80, 220, 120),  # slot 1 – green
    (200, 160,  50),  # slot 2 – yellow
]

CELL_SIZE    = 60
GRID_PADDING = 20
QUEUE_WIDTH  = 200
PANEL_PAD    = 18
INFO_HEIGHT  = 80
FONT_SIZE    = 20
TITLE_SIZE   = 28


def make_window() -> tuple[pygame.Surface, dict]:
    """Create and return the main surface plus layout metrics."""
    board_px = CELL_SIZE * param.BOARD_SIZE
    win_w = GRID_PADDING * 2 + board_px + QUEUE_WIDTH + PANEL_PAD * 2
    win_h = GRID_PADDING * 2 + board_px + INFO_HEIGHT

    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("Block Blast – Agent Demo")

    layout = {
        "board_x": GRID_PADDING,
        "board_y": GRID_PADDING,
        "board_px": board_px,
        "queue_x": GRID_PADDING + board_px + PANEL_PAD,
        "queue_y": GRID_PADDING,
        "info_y": GRID_PADDING + board_px + 10,
        "win_w": win_w,
        "win_h": win_h,
    }
    return screen, layout


def draw_board(
    surface: pygame.Surface,
    board: Board,
    x0: int,
    y0: int,
) -> None:
    """Draw the 8x8 board grid at pixel offset (x0, y0)."""
    board_px = CELL_SIZE * param.BOARD_SIZE
    # Background
    pygame.draw.rect(surface, GRID_BG, (x0, y0, board_px, board_px), border_radius=6)

    for row in range(param.BOARD_SIZE):
        for col in range(param.BOARD_SIZE):
            cx = x0 + col * CELL_SIZE
            cy = y0 + row * CELL_SIZE
            rect = pygame.Rect(cx + 2, cy + 2, CELL_SIZE - 4, CELL_SIZE - 4)
            if board.grid[row, col]:
                pygame.draw.rect(surface, CELL_COLOR, rect, border_radius=5)
                pygame.draw.rect(surface, CELL_BORDER, rect, width=2, border_radius=5)
                # Highlight top-left corner shine
                shine = pygame.Rect(rect.x + 4, rect.y + 4, rect.width // 3, rect.height // 5)
                pygame.draw.rect(surface, CELL_HIGHLIGHT, shine, border_radius=3)
            else:
                pygame.draw.rect(surface, GRID_LINE, rect, width=1, border_radius=3)


def draw_piece_preview(
    surface: pygame.Surface,
    piece: Piece,
    x0: int,
    y0: int,
    color: tuple[int, int, int],
    cell_size: int = 22,
) -> None:
    """Draw a small piece preview with `cell_size` pixels per cell."""
    for dr, dc in piece.cells:
        cx = x0 + dc * cell_size
        cy = y0 + dr * cell_size
        rect = pygame.Rect(cx + 1, cy + 1, cell_size - 2, cell_size - 2)
        pygame.draw.rect(surface, color, rect, border_radius=4)
        lighter = tuple(min(255, c + 50) for c in color)
        pygame.draw.rect(surface, lighter, rect, width=1, border_radius=4)


def draw_queue(
    surface: pygame.Surface,
    queue: list[Piece],
    x0: int,
    y0: int,
    win_h: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    """Draw the 3-piece queue panel."""
    panel_h = win_h - y0 - INFO_HEIGHT - GRID_PADDING
    pygame.draw.rect(
        surface, QUEUE_BG,
        (x0, y0, QUEUE_WIDTH - PANEL_PAD, panel_h),
        border_radius=8,
    )
    pygame.draw.rect(
        surface, QUEUE_BORDER,
        (x0, y0, QUEUE_WIDTH - PANEL_PAD, panel_h),
        width=2, border_radius=8,
    )

    label = small_font.render("NEXT PIECES", True, DIM_TEXT)
    surface.blit(label, (x0 + 10, y0 + 10))

    slot_h = (panel_h - 40) // param.QUEUE_SIZE
    for i, piece in enumerate(queue):
        sy = y0 + 40 + i * slot_h
        lbl = small_font.render(f"#{i + 1}", True, DIM_TEXT)
        surface.blit(lbl, (x0 + 10, sy))
        draw_piece_preview(
            surface, piece,
            x0 + 14, sy + 18,
            color=PIECE_COLORS[i % len(PIECE_COLORS)],
        )


def draw_info(
    surface: pygame.Surface,
    score: float,
    steps: int,
    game_over: bool,
    paused: bool,
    x0: int,
    y0: int,
    win_w: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    """Draw score / status bar at the bottom."""
    score_txt = font.render(f"Score: {int(score):,}", True, TEXT_COLOR)
    steps_txt = small_font.render(f"Steps: {steps}", True, DIM_TEXT)
    surface.blit(score_txt, (x0, y0))
    surface.blit(steps_txt, (x0, y0 + 30))

    if game_over:
        go_txt = font.render("GAME OVER  [R] restart", True, GAME_OVER_COL)
        surface.blit(go_txt, (x0 + 200, y0 + 5))
    elif paused:
        p_txt = font.render("PAUSED  [SPACE] resume", True, DIM_TEXT)
        surface.blit(p_txt, (x0 + 200, y0 + 5))
    else:
        hint = small_font.render("[SPACE] pause   [R] reset", True, DIM_TEXT)
        surface.blit(hint, (x0 + 200, y0 + 5))
