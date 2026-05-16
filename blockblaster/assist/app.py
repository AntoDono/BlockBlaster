"""Pygame application loop for the assist side-by-side viewer."""

from __future__ import annotations

from typing import Optional

import pygame

from blockblaster.assist.advisor import Advisor
from blockblaster.assist.calibration import CalibrationBox, CalibrationConfig
from blockblaster.assist.device_stream import DeviceStream
from blockblaster.assist.layout import (
    BG_COLOR,
    PHONE_RECT,
    RECON_RECT,
    STATUS_RECT,
    make_window,
)
from blockblaster.assist.piece_recognizer import PieceRecognizer
from blockblaster.assist.render import (
    bgr_to_surface,
    draw_drag_preview,
    draw_grid_overlay,
    draw_phone_panel,
    draw_queue_overlay,
    draw_recon_panel,
    draw_status_bar,
    draw_suggestion_on_phone,
)
from blockblaster.assist.scanner import scan_board
from blockblaster.game.board import Board

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
    queue_confidences: list[float] = []

    # Load persisted calibration (both slots)
    cfg = CalibrationConfig.load()

    # Piece recognizer (pre-computes 32 templates at init)
    recognizer = PieceRecognizer()

    # Move advisor (loads ValueNet from model.pt; degrades gracefully if missing)
    advisor = Advisor()
    if advisor.last_error:
        print(f"[assist] advisor disabled: {advisor.last_error}")

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

    stream = DeviceStream()
    stream.start()

    # Cached phone-panel surface (regenerated only when the frame changes
    # OR when the panel rect changes — neither happens often)
    phone_cache: Optional[tuple] = None  # (frame_id, surf, scale, blit_x, blit_y)
    last_processed_frame_id: int = -1

    running = True
    while running:
        frame, frame_id = stream.get_latest_with_id()
        frame_changed = frame is not None and frame_id != last_processed_frame_id

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

                elif event.key == pygame.K_d:
                    if frame is not None and cfg.queue is not None and cfg.queue.is_valid():
                        out = recognizer.save_debug(frame, cfg.queue)
                        print(f"Saved queue debug images to: {out.resolve()}")
                    else:
                        print("Debug skipped: need a frame and a calibrated queue box.")

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

        # ── Board scan + queue recognition (only on new frames) ─────────
        if frame_changed:
            if cfg.grid is not None and cfg.grid.is_valid():
                board.grid = scan_board(frame, cfg.grid)
            if cfg.queue is not None and cfg.queue.is_valid():
                results = recognizer.recognize_queue_with_confidence(frame, cfg.queue)
                queue = [p for p, _ in results]
                queue_confidences = [c for _, c in results]
            last_processed_frame_id = frame_id

        # ── Advisor (cached on board + queue) ────────────────────────────
        suggestion = advisor.suggest(board.grid, queue) if queue else None

        # ── Render ───────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Re-resize the phone frame only when the underlying frame changed
        if frame is not None and (
            phone_cache is None or phone_cache[0] != frame_id
        ):
            content_rect = pygame.Rect(
                PHONE_RECT.x + 4, PHONE_RECT.y + 30,
                PHONE_RECT.width - 8, PHONE_RECT.height - 38,
            )
            surf, scale, bx, by = bgr_to_surface(frame, content_rect)
            phone_cache = (frame_id, surf, scale, bx, by)

        cached = (
            (phone_cache[1], phone_cache[2], phone_cache[3], phone_cache[4])
            if phone_cache is not None else None
        )
        _scale, _blit_x, _blit_y = draw_phone_panel(
            screen,
            frame=frame,
            rect=PHONE_RECT,
            error_msg=stream.last_error,
            font=font,
            small_font=small_font,
            cached_surface=cached,
        )

        # Calibration overlays on top of the phone panel
        if frame is not None:
            if cfg.grid is not None:
                draw_grid_overlay(screen, cfg.grid, _scale, _blit_x, _blit_y)
            if cfg.queue is not None:
                draw_queue_overlay(
                    screen, cfg.queue, _scale, _blit_x, _blit_y, small_font,
                    chosen_slot=suggestion.slot if suggestion is not None else None,
                )
            if suggestion is not None and cfg.grid is not None and cfg.grid.is_valid():
                draw_suggestion_on_phone(
                    screen, cfg.grid, suggestion,
                    _scale, _blit_x, _blit_y,
                )

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
            suggestion=suggestion,
            queue_confidences=queue_confidences,
        )

        # Build status hint showing active mode
        mode_label = "GRID" if calib_mode == MODE_GRID else "PIECES"
        hint = (
            f"[Tab] mode: {mode_label}   "
            "[drag] set box   "
            "[R] clear   "
            "[D] dump queue debug   "
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
        clock.tick()  # uncapped — render as fast as possible

    stream.stop()
    pygame.quit()
