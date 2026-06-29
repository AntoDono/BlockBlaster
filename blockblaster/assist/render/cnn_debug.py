"""CNN input/output debug panel for the assist GUI."""

from __future__ import annotations

import pygame

from blockblaster.assist.vision.analyzer import ReconSnapshot
from blockblaster.assist.render.phone import (
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

_INPUT_SZ     = 52
_PREVIEW_CELL = 11
_ARROW_W      = 18
_ARROW_GAP    = 6
_SLOT_H       = 110
_TEXT_GAP     = 2


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

    content = pygame.Rect(rect.x + 4, rect.y + 28, rect.width - 8, rect.height - 34)
    if not snap.pieces:
        msg = small_font.render("no pieces detected", True, DIM_TEXT)
        screen.blit(msg, (content.centerx - msg.get_width() // 2,
                          content.centery - msg.get_height() // 2))
        return

    for i, pd in enumerate(snap.pieces):
        slot_rect = pygame.Rect(content.x, content.y + i * _SLOT_H, content.width, _SLOT_H)
        _draw_slot(screen, slot_rect, i, pd, small_font)
        if i + 1 < len(snap.pieces):
            pygame.draw.line(
                screen, PANEL_BORDER,
                (slot_rect.x + 6, slot_rect.bottom - 1),
                (slot_rect.right - 6, slot_rect.bottom - 1),
            )


def _piece_label(pd) -> str:
    if pd.piece is None:
        return "—"
    return pd.piece.name


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
        out_w = out_h = _PREVIEW_CELL * 2

    row_w = _INPUT_SZ + _ARROW_GAP + _ARROW_W + _ARROW_GAP + out_w
    row_visual_h = max(_INPUT_SZ, out_h)

    tag = font.render(f"#{slot + 1}", True, LABEL_COL)
    in_lbl = font.render("in", True, DIM_TEXT)
    conf_txt = font.render(f"{pd.confidence:.2f}", True, _conf_color(pd.confidence))
    piece_txt = font.render(_piece_label(pd), True, SUGGEST_FILL if pd.piece else DIM_TEXT)

    lh = font.get_height()
    block_h = (
        tag.get_height() + _TEXT_GAP
        + row_visual_h + _TEXT_GAP
        + max(in_lbl.get_height(), conf_txt.get_height() + _TEXT_GAP + piece_txt.get_height())
    )
    y = rect.y + (rect.height - block_h) // 2
    row_x = rect.centerx - row_w // 2

    screen.blit(tag, (rect.centerx - tag.get_width() // 2, y))
    y += tag.get_height() + _TEXT_GAP

    input_x = row_x
    preview_x = row_x + _INPUT_SZ + _ARROW_GAP + _ARROW_W + _ARROW_GAP
    row_mid_y = y + row_visual_h // 2

    input_y = row_mid_y - _INPUT_SZ // 2
    preview_y = row_mid_y - out_h // 2

    input_rect = pygame.Rect(input_x, input_y, _INPUT_SZ, _INPUT_SZ)
    if pd.cnn_input is not None:
        surf, _, bx, by = bgr_to_surface(pd.cnn_input, input_rect)
        screen.blit(surf, (bx, by))
    else:
        pygame.draw.rect(screen, (35, 35, 50), input_rect, border_radius=2)

    screen.blit(
        in_lbl,
        (input_rect.centerx - in_lbl.get_width() // 2, input_rect.bottom + 1),
    )

    _draw_arrow(screen, input_rect.right + _ARROW_GAP, row_mid_y)

    if pd.piece is not None:
        draw_piece_preview(
            screen, pd.piece, preview_x, preview_y, color=color, cell_size=_PREVIEW_CELL,
        )
    else:
        dash = font.render("?", True, DIM_TEXT)
        screen.blit(
            dash,
            (preview_x + out_w // 2 - dash.get_width() // 2, preview_y + out_h // 2),
        )

    preview_cx = preview_x + out_w // 2
    meta_y = y + row_visual_h + _TEXT_GAP
    screen.blit(conf_txt, (preview_cx - conf_txt.get_width() // 2, meta_y))
    screen.blit(
        piece_txt,
        (preview_cx - piece_txt.get_width() // 2, meta_y + lh + _TEXT_GAP),
    )


def _draw_arrow(screen: pygame.Surface, x: int, y: int) -> None:
    col = DIM_TEXT
    pygame.draw.line(screen, col, (x, y), (x + _ARROW_W, y), 2)
    pygame.draw.polygon(
        screen, col,
        [(x + _ARROW_W, y), (x + _ARROW_W - 5, y - 4), (x + _ARROW_W - 5, y + 4)],
    )


def _conf_color(conf: float) -> tuple[int, int, int]:
    if conf >= 0.90:
        return _CONF_OK
    if conf >= 0.70:
        return _CONF_MID
    return _CONF_LOW
