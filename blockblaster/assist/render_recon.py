"""Reconstructed-scene panel rendering for the assist GUI."""

from __future__ import annotations

from typing import Optional

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

    draw_board(screen, board, board_x, board_y)

    if suggestion is not None:
        _draw_ghost_piece_on_board(screen, suggestion, board_x, board_y)

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
