"""Spatial reconstruction of the phone screen for the assist GUI.

Renders a phone-aspect canvas inside the recon panel and places:
  * the detected board as an 8x8 grid coloured by the scanned occupancy,
  * each detected tray piece rendered at its detected on-screen position,
  * the advisor's suggested placement as a ghost on the board.
"""

from __future__ import annotations

from typing import Optional

import pygame

from blockblaster.assist.analyzer import PieceDetection, ReconSnapshot
from blockblaster.assist.render_phone import (
    DIM_TEXT,
    LABEL_COL,
    PANEL_BG,
    PANEL_BORDER,
    SUGGEST_BORDER,
    SUGGEST_FILL,
    SUGGEST_FILL_A,
)
from blockblaster.gui.render import (
    CELL_BORDER,
    CELL_COLOR,
    GRID_BG,
    GRID_LINE,
    PIECE_COLORS,
)

BOARD_SIZE     = 8
CANVAS_BG      = (12, 12, 22)
CANVAS_BORDER  = (40, 40, 60)


def draw_recon_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    snap: ReconSnapshot,
    frame_w: int,
    frame_h: int,
    small_font: pygame.font.Font,
) -> None:
    pygame.draw.rect(screen, PANEL_BG,     rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    lbl = small_font.render("RECONSTRUCTED SCENE", True, LABEL_COL)
    screen.blit(lbl, (rect.x + 10, rect.y + 8))

    content = pygame.Rect(rect.x + 4, rect.y + 30, rect.width - 8, rect.height - 38)
    if frame_w <= 0 or frame_h <= 0:
        _draw_placeholder(screen, content, small_font, "Waiting for frame…")
        _draw_caption(screen, rect, small_font, snap.suggestion is not None)
        return

    canvas = _fit_phone_canvas(content, frame_w, frame_h)
    pygame.draw.rect(screen, CANVAS_BG,     canvas, border_radius=12)
    pygame.draw.rect(screen, CANVAS_BORDER, canvas, width=2, border_radius=12)

    sx = canvas.width  / frame_w
    sy = canvas.height / frame_h

    if snap.board_bbox is not None:
        _draw_board(screen, canvas, sx, sy, snap)

    suggestion_slot = snap.suggestion.slot if snap.suggestion is not None else None
    for i, pd in enumerate(snap.pieces):
        _draw_piece(screen, canvas, sx, sy, pd, i, highlight=(i == suggestion_slot))

    _draw_caption(screen, rect, small_font, snap.suggestion is not None, snap)


def _fit_phone_canvas(area: pygame.Rect, frame_w: int, frame_h: int) -> pygame.Rect:
    scale = min(area.width / frame_w, area.height / frame_h)
    w = int(frame_w * scale)
    h = int(frame_h * scale)
    x = area.x + (area.width  - w) // 2
    y = area.y + (area.height - h) // 2
    return pygame.Rect(x, y, w, h)


def _bbox_to_rect(
    canvas: pygame.Rect,
    sx: float, sy: float,
    bbox: tuple[int, int, int, int],
) -> pygame.Rect:
    x, y, w, h = bbox
    return pygame.Rect(
        canvas.x + int(x * sx),
        canvas.y + int(y * sy),
        max(1, int(w * sx)),
        max(1, int(h * sy)),
    )


def _draw_board(
    screen: pygame.Surface,
    canvas: pygame.Rect,
    sx: float, sy: float,
    snap: ReconSnapshot,
) -> None:
    rect = _bbox_to_rect(canvas, sx, sy, snap.board_bbox)
    pygame.draw.rect(screen, GRID_BG, rect, border_radius=4)

    cw = rect.width  / BOARD_SIZE
    ch = rect.height / BOARD_SIZE
    r  = _cell_radius(cw, ch)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            cell = pygame.Rect(
                int(rect.x + col * cw),
                int(rect.y + row * ch),
                max(1, int(cw)),
                max(1, int(ch)),
            )
            inner = cell.inflate(-2, -2)
            if snap.board_grid[row, col]:
                pygame.draw.rect(screen, CELL_COLOR, inner, border_radius=r)
                pygame.draw.rect(screen, CELL_BORDER, inner, width=1, border_radius=r)
            else:
                pygame.draw.rect(screen, GRID_LINE, inner, width=1, border_radius=r)

    if snap.suggestion is not None:
        _draw_ghost(screen, rect, cw, ch, snap.suggestion)


def _draw_ghost(
    screen: pygame.Surface,
    board_rect: pygame.Rect,
    cw: float, ch: float,
    suggestion,
) -> None:
    for dr, dc in suggestion.piece.cells:
        r, c = suggestion.row + dr, suggestion.col + dc
        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            continue
        cell = pygame.Rect(
            int(board_rect.x + c * cw),
            int(board_rect.y + r * ch),
            max(1, int(cw)),
            max(1, int(ch)),
        )
        inner = cell.inflate(-2, -2)
        ov = pygame.Surface(inner.size, pygame.SRCALPHA)
        ov.fill((*SUGGEST_FILL, SUGGEST_FILL_A))
        screen.blit(ov, inner.topleft)
        pygame.draw.rect(screen, SUGGEST_BORDER, inner, width=2, border_radius=_cell_radius(cw, ch))


def _draw_piece(
    screen: pygame.Surface,
    canvas: pygame.Rect,
    sx: float, sy: float,
    pd: PieceDetection,
    slot: int,
    highlight: bool,
) -> None:
    rect  = _bbox_to_rect(canvas, sx, sy, pd.bbox)
    color = SUGGEST_FILL if highlight else PIECE_COLORS[slot % len(PIECE_COLORS)]

    if pd.piece is None:
        pygame.draw.rect(screen, CANVAS_BORDER, rect, width=1, border_radius=4)
        return

    rows, cols = pd.piece.rows, pd.piece.cols
    cw = rect.width  / cols
    ch = rect.height / rows
    r  = _cell_radius(cw, ch)
    for dr, dc in pd.piece.cells:
        cell = pygame.Rect(
            int(rect.x + dc * cw),
            int(rect.y + dr * ch),
            max(1, int(cw)),
            max(1, int(ch)),
        )
        inner = cell.inflate(-2, -2)
        pygame.draw.rect(screen, color, inner, border_radius=r)
        lighter = tuple(min(255, c + 50) for c in color)
        pygame.draw.rect(screen, lighter, inner, width=1, border_radius=r)


def _cell_radius(cw: float, ch: float) -> int:
    return max(0, min(2, int(min(cw, ch) // 6)))


def _draw_placeholder(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    text: str,
) -> None:
    s = font.render(text, True, DIM_TEXT)
    screen.blit(s, (rect.centerx - s.get_width() // 2,
                    rect.centery - s.get_height() // 2))


def _draw_caption(
    screen: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    have_suggestion: bool,
    snap: Optional[ReconSnapshot] = None,
) -> None:
    if have_suggestion and snap is not None and snap.suggestion is not None:
        s = snap.suggestion
        text = (
            f"suggested: {s.piece.name} at "
            f"row {s.row + 1}, col {s.col + 1} (slot {s.slot + 1})"
        )
        col = SUGGEST_FILL
    else:
        text = "scanning…"
        col  = DIM_TEXT
    label = font.render(text, True, col)
    screen.blit(label, (rect.centerx - label.get_width() // 2, rect.bottom - 22))
