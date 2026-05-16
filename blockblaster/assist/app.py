"""Pygame application loop for the assist side-by-side viewer."""

from __future__ import annotations

from typing import Optional

import pygame

from blockblaster.assist.calibration import CalibrationBox, CalibrationConfig
from blockblaster.assist.device_stream import DeviceStream
from blockblaster.assist.layout import (
    BG_COLOR,
    PHONE_RECT,
    RECON_RECT,
    STATUS_RECT,
    make_window,
)
from blockblaster.assist.render import (
    draw_drag_preview,
    draw_grid_overlay,
    draw_phone_panel,
    draw_queue_overlay,
    draw_recon_panel,
    draw_status_bar,
)
from blockblaster.assist.piece_recognizer import PieceRecognizer
from blockblaster.assist.scanner import scan_board
from blockblaster.game.board import Board

FPS = 30

# Calibration modes
MODE_GRID   = "grid"
MODE_PIECES = "pieces"


def run() -> None:
    """Launch the assist viewer window.

    Left panel:  live iOS phone screen (via tunneld/DVT).
    Right panel: reconstructed Block Blast scene from the scanned grid.

    Controls:
        Tab                 – toggle calibration mode (GRID / PIECES)
        Drag on left panel  – set bounding box for the active mode
        R                   – clear the active mode's box
        Q / ESC             – quit
    """
    pygame.init()
    pygame.font.init()

    font       = pygame.font.SysFont("monospace", 20, bold=True)
    small_font = pygame.font.SysFont("monospace", 15)

    screen = make_window()
    clock  = pygame.time.Clock()

    board = Board()
    queue: list = []

    # Load persisted calibration (both slots)
    cfg = CalibrationConfig.load()

    # Active calibration mode
    calib_mode: str = MODE_GRID

    # Drag state
    drag_start: Optional[tuple[int, int]] = None
    drag_cur:   Optional[tuple[int, int]] = None

    # Last render info for coordinate mapping
    _scale:   float = 1.0
    _blit_x:  int   = 0
    _blit_y:  int   = 0
    _frame_w: int   = 1
    _frame_h: int   = 1

    recognizer = PieceRecognizer()

    stream = DeviceStream()
    stream.start()

    running = True
    while running:
        frame = stream.get_latest()

        if frame is not None:
            _frame_h, _frame_w = frame.shape[:2]

        # ── Events ───────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                elif event.key == pygame.K_TAB:
                    calib_mode = MODE_PIECES if calib_mode == MODE_GRID else MODE_GRID

                elif event.key == pygame.K_r:
                    if calib_mode == MODE_GRID:
                        cfg.grid = None
                        board = Board()
                    else:
                        cfg.queue = None
                    cfg.save()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if PHONE_RECT.collidepoint(mx, my) and frame is not None:
                    drag_start = (mx, my)
                    drag_cur   = (mx, my)

            elif event.type == pygame.MOUSEMOTION:
                if drag_start is not None:
                    drag_cur = event.pos

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if drag_start is not None and drag_cur is not None and frame is not None:
                    new_box = CalibrationBox.from_screen(
                        drag_start[0], drag_start[1],
                        drag_cur[0],   drag_cur[1],
                        _scale, _blit_x, _blit_y,
                        _frame_w, _frame_h,
                    )
                    if new_box.is_valid():
                        if calib_mode == MODE_GRID:
                            cfg.grid = new_box
                        else:
                            cfg.queue = new_box
                        cfg.save()
                drag_start = None
                drag_cur   = None

        # ── Board scan ───────────────────────────────────────────────────
        if frame is not None and cfg.grid is not None and cfg.grid.is_valid():
            board.grid = scan_board(frame, cfg.grid)

        # ── Queue scan ───────────────────────────────────────────────────
        if frame is not None and cfg.queue is not None and cfg.queue.is_valid():
            recognized = recognizer.recognize_queue(frame, cfg.queue)
            queue = [p for p in recognized if p is not None]

        # ── Render ───────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        _scale, _blit_x, _blit_y = draw_phone_panel(
            screen,
            frame=frame,
            rect=PHONE_RECT,
            error_msg=stream.last_error,
            font=font,
            small_font=small_font,
        )

        # Calibration overlays on top of the phone panel
        if frame is not None:
            if cfg.grid is not None:
                draw_grid_overlay(screen, cfg.grid, _scale, _blit_x, _blit_y)
            if cfg.queue is not None:
                draw_queue_overlay(screen, cfg.queue, _scale, _blit_x, _blit_y, small_font)

        # Drag-in-progress preview
        if drag_start is not None and drag_cur is not None:
            draw_drag_preview(screen, drag_start, drag_cur)

        draw_recon_panel(
            screen,
            rect=RECON_RECT,
            board=board,
            queue=queue,
            font=font,
            small_font=small_font,
        )

        # Build status hint showing active mode
        mode_label = "GRID" if calib_mode == MODE_GRID else "PIECES"
        hint = (
            f"[Tab] mode: {mode_label}   "
            "[drag] set box   "
            "[R] clear   "
            "[Q/ESC] quit"
        )
        draw_status_bar(
            screen,
            fps=clock.get_fps(),
            has_device=frame is not None,
            rect=STATUS_RECT,
            small_font=small_font,
            hint=hint,
        )

        pygame.display.flip()
        clock.tick(FPS)

    stream.stop()
    pygame.quit()
