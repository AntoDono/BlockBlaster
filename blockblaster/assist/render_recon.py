"""Reconstructed-scene panel rendering for the assist GUI."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.render_phone import (
    DIM_TEXT,
    LABEL_COL,
    PANEL_BG,
    PANEL_BORDER,
    SUGGEST_BORDER,
    SUGGEST_FILL,
    SUGGEST_FILL_A,
)
from blockblaster.game.board import Board
from blockblaster.game.pieces import Piece
from blockblaster.gui.render import (
    CELL_SIZE,
    PIECE_COLORS,
    QUEUE_WIDTH,
    draw_board,
    draw_piece_preview,
)


def draw_recon_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    board: Board,
    queue: list[Piece],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    suggestion: Optional[Suggestion] = None,
    queue_confidences: Optional[list[float]] = None,
    detection: Optional[tuple[int, int, int, int, float]] = None,
    debug_mask: Optional[np.ndarray] = None,
    debug_view: bool = False,
) -> None:
    """Draw the right reconstructed game-state panel."""
    pygame.draw.rect(screen, PANEL_BG,     rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    lbl = small_font.render("RECONSTRUCTED SCENE", True, LABEL_COL)
    screen.blit(lbl, (rect.x + 10, rect.y + 8))

    board_px        = CELL_SIZE * 8
    queue_panel_w   = QUEUE_WIDTH
    total_inner_w   = board_px + 12 + queue_panel_w
    board_x         = rect.x + (rect.width - total_inner_w) // 2
    board_y         = rect.y + 36 + (rect.height - 36 - board_px) // 2

    if debug_view and debug_mask is not None:
        _draw_motion_mask(screen, debug_mask, board_x, board_y, board_px, small_font)
    else:
        draw_board(screen, board, board_x, board_y)

    if suggestion is not None:
        _draw_ghost_piece_on_board(screen, suggestion, board_x, board_y)

    if detection is not None:
        piece_for_overlay = suggestion.piece if suggestion is not None else None
        _draw_detection_overlay(
            screen, detection, board_x, board_y, small_font, piece_for_overlay,
        )

    qx          = board_x + board_px + 12
    chosen_slot = suggestion.slot if suggestion is not None else None
    _draw_mini_queue(
        screen, queue, qx, board_y, board_px, small_font,
        chosen_slot=chosen_slot,
        confidences=queue_confidences,
    )

    if suggestion is not None:
        caption_text = (
            f"suggested: {suggestion.piece.name} at "
            f"row {suggestion.row + 1}, col {suggestion.col + 1} "
            f"(slot {suggestion.slot + 1})"
        )
        caption_col = SUGGEST_FILL
    else:
        caption_text = "drag on phone panel to calibrate grid"
        caption_col  = DIM_TEXT
    caption = small_font.render(caption_text, True, caption_col)
    screen.blit(caption, (rect.centerx - caption.get_width() // 2, rect.bottom - 26))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _confidence_color(conf: float) -> tuple[int, int, int]:
    """Traffic-light colour: green ≥ 0.90, yellow 0.70–0.90, red < 0.70."""
    if conf >= 0.90:
        return (90, 220, 110)
    if conf >= 0.70:
        return (240, 210, 80)
    return (235, 90, 90)


def _draw_mini_queue(
    surface: pygame.Surface,
    queue: list[Piece],
    x0: int,
    y0: int,
    panel_h: int,
    small_font: pygame.font.Font,
    chosen_slot: Optional[int] = None,
    confidences: Optional[list[float]] = None,
) -> None:
    """Minimal queue preview in the recon panel.

    Each slot shows a ``p=0.XX`` confidence badge colour-coded
    green / yellow / red next to the ``#N`` label.
    """
    from blockblaster.gui.render import DIM_TEXT as GUI_DIM, QUEUE_BG, QUEUE_BORDER

    qw = QUEUE_WIDTH - 10
    pygame.draw.rect(surface, QUEUE_BG,     (x0, y0, qw, panel_h), border_radius=8)
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

        if i == chosen_slot:
            slot_rect = pygame.Rect(x0 + 4, sy - 4, qw - 8, slot_h - 2)
            ov = pygame.Surface(slot_rect.size, pygame.SRCALPHA)
            ov.fill((*SUGGEST_FILL, 60))
            surface.blit(ov, slot_rect.topleft)
            pygame.draw.rect(surface, SUGGEST_BORDER, slot_rect, width=2, border_radius=6)

        label_col = SUGGEST_FILL if i == chosen_slot else GUI_DIM
        num = small_font.render(f"#{i + 1}", True, label_col)
        surface.blit(num, (x0 + 8, sy))

        if confidences is not None and i < len(confidences):
            conf_s = small_font.render(f"p={confidences[i]:.2f}", True,
                                       _confidence_color(confidences[i]))
            surface.blit(conf_s, (x0 + qw - conf_s.get_width() - 8, sy))

        if piece is None:
            dash = small_font.render("—", True, GUI_DIM)
            surface.blit(dash, (x0 + qw // 2 - dash.get_width() // 2, sy + 18))
            continue

        piece_color = SUGGEST_FILL if i == chosen_slot else PIECE_COLORS[i % len(PIECE_COLORS)]
        draw_piece_preview(surface, piece, x0 + 10, sy + 18, color=piece_color)


def _draw_motion_mask(
    surface: pygame.Surface,
    mask: np.ndarray,
    board_x: int,
    board_y: int,
    board_px: int,
    small_font: pygame.font.Font,
) -> None:
    """Render the servo's motion mask in place of the recon board.

    Moved pixels render bright cyan on near-black so the matcher's
    view is unmistakable.  An 8x8 grid is drawn on top so the cell
    structure stays legible.
    """
    if mask.ndim != 2 or mask.size == 0:
        return

    h, w = mask.shape
    # Build an RGB image: moved pixels cyan, static near-black.
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    moved = mask > 0
    rgb[..., 0][moved] = 0
    rgb[..., 1][moved] = 220
    rgb[..., 2][moved] = 255
    rgb[~moved] = (16, 18, 28)

    # pygame surfarray expects (w, h, 3) so swap axes.
    surf = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
    surf = pygame.transform.smoothscale(surf, (board_px, board_px))
    surface.blit(surf, (board_x, board_y))

    cell = board_px / 8
    grid_col = (60, 70, 90)
    for i in range(9):
        x = board_x + int(i * cell)
        y = board_y + int(i * cell)
        pygame.draw.line(surface, grid_col, (x, board_y), (x, board_y + board_px), 1)
        pygame.draw.line(surface, grid_col, (board_x, y), (board_x + board_px, y), 1)

    badge = small_font.render(
        f"motion mask {w}x{h}", True, (200, 230, 240),
    )
    badge_bg = pygame.Surface(
        (badge.get_width() + 8, badge.get_height() + 4), pygame.SRCALPHA,
    )
    badge_bg.fill((0, 0, 0, 160))
    surface.blit(badge_bg, (board_x + 4, board_y + board_px - badge.get_height() - 8))
    surface.blit(badge, (board_x + 8, board_y + board_px - badge.get_height() - 6))


_DETECT_FILL    = (255, 0, 200)   # hot magenta — contrasts hard against the
                                  # blue suggestion ghost and the board's
                                  # navy background, so the live detection
                                  # never blends into either.
_DETECT_FILL_A  = 120
_DETECT_BORDER  = (255, 80, 230)


def _draw_detection_overlay(
    surface: pygame.Surface,
    detection: tuple[int, int, int, int, float],
    board_x: int,
    board_y: int,
    small_font: pygame.font.Font,
    piece: Optional[Piece],
) -> None:
    """Paint the live matcher detection on the recon board.

    Renders the piece's actual cell pattern when available, falling back
    to the matched bounding rectangle otherwise.  Adds a ``score=0.NN``
    badge in the cyan piece colour.
    """
    tl_col, tl_row, p_rows, p_cols, score = detection

    if piece is not None:
        cells = list(piece.cells)
    else:
        cells = [(dr, dc) for dr in range(p_rows) for dc in range(p_cols)]

    for dr, dc in cells:
        r, c = tl_row + dr, tl_col + dc
        if not (0 <= r < 8 and 0 <= c < 8):
            continue
        rect = pygame.Rect(
            board_x + c * CELL_SIZE + 2,
            board_y + r * CELL_SIZE + 2,
            CELL_SIZE - 4, CELL_SIZE - 4,
        )
        ov = pygame.Surface(rect.size, pygame.SRCALPHA)
        ov.fill((*_DETECT_FILL, _DETECT_FILL_A))
        surface.blit(ov, rect.topleft)
        pygame.draw.rect(surface, _DETECT_BORDER, rect, width=4, border_radius=5)

    badge_text = f"DET score={score:.2f}"
    badge = small_font.render(badge_text, True, _DETECT_BORDER)
    badge_bg = pygame.Surface(
        (badge.get_width() + 8, badge.get_height() + 4), pygame.SRCALPHA,
    )
    badge_bg.fill((0, 0, 0, 160))
    surface.blit(badge_bg, (board_x + 4, board_y + 4))
    surface.blit(badge, (board_x + 8, board_y + 6))


def _draw_ghost_piece_on_board(
    surface: pygame.Surface,
    suggestion: Suggestion,
    board_x: int,
    board_y: int,
) -> None:
    for dr, dc in suggestion.piece.cells:
        r, c = suggestion.row + dr, suggestion.col + dc
        if not (0 <= r < 8 and 0 <= c < 8):
            continue
        rect = pygame.Rect(board_x + c * CELL_SIZE + 2, board_y + r * CELL_SIZE + 2,
                           CELL_SIZE - 4, CELL_SIZE - 4)
        ov = pygame.Surface(rect.size, pygame.SRCALPHA)
        ov.fill((*SUGGEST_FILL, SUGGEST_FILL_A))
        surface.blit(ov, rect.topleft)
        pygame.draw.rect(surface, SUGGEST_BORDER, rect, width=3, border_radius=5)
