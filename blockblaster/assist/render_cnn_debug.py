"""CNN input/output debug panel for the assist GUI."""

from __future__ import annotations

import pygame

from blockblaster.assist.analyzer import ReconSnapshot
from blockblaster.assist.render_phone import (
    DIM_TEXT,
    LABEL_COL,
    PANEL_BG,
    PANEL_BORDER,
    SUGGEST_FILL,
    bgr_to_surface,
)
from blockblaster.gui.render import PIECE_COLORS, draw_piece_preview

_CONF_OK  = (90, 220, 110)
_CONF_MID = (240, 210, 80)
_CONF_LOW = (235, 90, 90)

_INPUT_SZ     = 80
_PREVIEW_CELL = 16
_ARROW_W      = 32
_ARROW_GAP    = 10


def draw_cnn_debug_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    snap: ReconSnapshot,
    small_font: pygame.font.Font,
) -> None:
    pygame.draw.rect(screen, PANEL_BG,     rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    lbl = small_font.render("PIECE CNN", True, LABEL_COL)
    screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.y + 8))

    content = pygame.Rect(rect.x + 4, rect.y + 30, rect.width - 8, rect.height - 38)
    if not snap.pieces:
        msg = small_font.render("no pieces detected", True, DIM_TEXT)
        screen.blit(msg, (content.centerx - msg.get_width() // 2,
                            content.centery - msg.get_height() // 2))
        return

    slot_h = content.height // max(len(snap.pieces), 1)
    for i, pd in enumerate(snap.pieces):
        slot_rect = pygame.Rect(content.x, content.y + i * slot_h, content.width, slot_h)
        _draw_slot(screen, slot_rect, i, pd, small_font)


def _draw_slot(
    screen: pygame.Surface,
    rect: pygame.Rect,
    slot: int,
    pd,
    font: pygame.font.Font,
) -> None:
    color = PIECE_COLORS[slot % len(PIECE_COLORS)]
    if pd.piece is not None:
        out_w = pd.piece.cols * _PREVIEW_CELL
        out_h = pd.piece.rows * _PREVIEW_CELL
    else:
        out_w = out_h = _INPUT_SZ // 2

    row_w = _INPUT_SZ + _ARROW_GAP + _ARROW_W + _ARROW_GAP + out_w
    row_h = _INPUT_SZ + 14

    num = font.render(f"#{slot + 1}", True, LABEL_COL)
    if pd.piece is not None:
        piece_txt = font.render(pd.piece.name, True, SUGGEST_FILL)
    else:
        piece_txt = font.render("—", True, DIM_TEXT)
    conf_txt = font.render(f"p={pd.confidence:.2f}", True, _conf_color(pd.confidence))
    meta_h = piece_txt.get_height() + 4 + conf_txt.get_height()

    block_h = num.get_height() + 8 + row_h + 8 + meta_h
    cy = rect.y + (rect.height - block_h) // 2
    cx = rect.x + (rect.width - row_w) // 2

    screen.blit(num, (rect.centerx - num.get_width() // 2, cy))
    cy += num.get_height() + 8

    input_x = cx
    input_rect = pygame.Rect(input_x, cy, _INPUT_SZ, _INPUT_SZ)
    if pd.cnn_input is not None:
        surf, _, bx, by = bgr_to_surface(pd.cnn_input, input_rect)
        screen.blit(surf, (bx, by))
    else:
        pygame.draw.rect(screen, (35, 35, 50), input_rect, border_radius=2)

    in_lbl = font.render("input", True, DIM_TEXT)
    screen.blit(in_lbl, (input_rect.centerx - in_lbl.get_width() // 2, input_rect.bottom + 2))

    arrow_x = input_rect.right + _ARROW_GAP
    _draw_arrow(screen, arrow_x, input_rect.centery)

    out_x = arrow_x + _ARROW_W + _ARROW_GAP
    if pd.piece is not None:
        draw_piece_preview(screen, pd.piece, out_x, cy, color=color, cell_size=_PREVIEW_CELL)
    else:
        dash = font.render("?", True, DIM_TEXT)
        screen.blit(dash, (out_x + out_w // 2 - dash.get_width() // 2, cy + out_h // 2))

    out_lbl = font.render("detected", True, DIM_TEXT)
    screen.blit(out_lbl, (out_x + out_w // 2 - out_lbl.get_width() // 2, input_rect.bottom + 2))

    meta_y = cy + row_h + 8
    screen.blit(piece_txt, (rect.centerx - piece_txt.get_width() // 2, meta_y))
    screen.blit(conf_txt, (rect.centerx - conf_txt.get_width() // 2,
                           meta_y + piece_txt.get_height() + 4))


def _draw_arrow(screen: pygame.Surface, x: int, y: int) -> None:
    col = DIM_TEXT
    pygame.draw.line(screen, col, (x, y), (x + _ARROW_W, y), 2)
    pygame.draw.polygon(screen, col, [(x + _ARROW_W, y), (x + _ARROW_W - 6, y - 5), (x + _ARROW_W - 6, y + 5)])


def _conf_color(conf: float) -> tuple[int, int, int]:
    if conf >= 0.90:
        return _CONF_OK
    if conf >= 0.70:
        return _CONF_MID
    return _CONF_LOW
