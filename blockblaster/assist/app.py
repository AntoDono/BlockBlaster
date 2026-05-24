"""Pygame application loop for the assist side-by-side viewer.

The run loop here is intentionally narrow:

* Event handling lives in :mod:`app_events`.
* Auto-play move execution lives in :mod:`app_autoplay`.
* Controls panel, calibration status overlay, and status bar live in :mod:`app_overlay`.
* All mutable session state is bundled into :class:`AppState`.
"""

from __future__ import annotations

import time
from typing import Optional

import pygame

from blockblaster.assist.advisor import Advisor
from blockblaster.assist.analyzer import AnalysisWorker
from blockblaster.assist.app_autoplay import maybe_execute_auto_swipe
from blockblaster.assist.app_events import dispatch_event
from blockblaster.assist.app_overlay import draw_controls_panel
from blockblaster.assist.app_state import AppState
from blockblaster.assist.calibration import CalibrationConfig
from blockblaster.assist.layout import (
    BG_COLOR,
    CONTROLS_RECT,
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
    draw_servo_error_on_phone,
    draw_status_bar,
    draw_suggestion_on_phone,
    draw_swipe_arrow_on_phone,
)
from blockblaster.control.device import Device
from blockblaster.game.board import Board

_TARGET_FPS = 60   # render-thread cap; analysis runs on its own worker thread


def run(
    device: Optional[Device] = None,
    platform: Optional[str] = None,
    auto_play: bool = False,
) -> None:
    """Launch the assist viewer window.

    Left panel:  live phone screen (iOS DVT or Android ADB).
    Right panel: reconstructed Block Blast scene + AI suggestion.

    Parameters
    ----------
    device:
        Optional pre-built device.  Defaults to iOS read-only.
    platform:
        ``"ios"`` or ``"android"`` — selects platform-specific calibration file.
    auto_play:
        Ignored — auto-play is always off when the GUI opens.  Use the
        **Auto-play** chip or press ``A`` to enable it after launch.

    Controls
    --------
        Tab                 – toggle calibration mode (GRID / PIECES)
        A                   – toggle auto-play on/off
        Drag on left panel  – set bounding box for the active mode
        R                   – clear the active mode's box
        D                   – dump per-slot debug images
        S                   – save a screenshot of the window to
                              ``screenshots/`` in the project root
        Q / ESC             – quit

    The chip buttons at the bottom of the window mirror every keyboard
    shortcut; click them instead of pressing keys.
    """
    pygame.init()
    pygame.font.init()

    font       = pygame.font.SysFont("monospace", 20, bold=True)
    small_font = pygame.font.SysFont("monospace", 15)

    screen = make_window()
    clock  = pygame.time.Clock()

    # ── Long-lived collaborators ────────────────────────────────────────
    cfg        = CalibrationConfig.load(platform=platform)
    recognizer = PieceRecognizer()
    advisor    = Advisor()
    if advisor.last_error:
        print(f"[assist] advisor disabled: {advisor.last_error}")

    if device is None:
        from blockblaster.control.ios_readonly import IosReadOnlyDevice
        device = IosReadOnlyDevice()
    stream = device
    stream.start()

    analyzer = AnalysisWorker(
        device=stream, cfg=cfg, recognizer=recognizer, advisor=advisor,
    )
    analyzer.start()

    # ── Session state ───────────────────────────────────────────────────
    state = AppState(
        cfg=cfg,
        platform=platform,
        board=Board(),
    )
    # Auto-play is always OFF on startup — user must click the chip or press A.
    state.auto_enabled = False
    if auto_play:
        print("[assist] auto_play flag is ignored — click Auto-play or press A to enable.")

    queue: list = []
    queue_confidences: list[float] = []

    # Caches keyed by content — avoid re-blitting / re-rendering identical frames
    phone_cache: Optional[tuple] = None  # (frame_id, surf, scale, blit_x, blit_y)
    recon_cache_key:  Optional[tuple] = None
    recon_cache_surf: Optional[pygame.Surface] = None
    last_analysis_frame_id: int = -1

    # ADB / device FPS counter — counts unique frame_id changes per second
    _adb_fps_window_start: float = time.monotonic()
    _adb_frame_count: int = 0
    _adb_fps: float = 0.0
    _prev_frame_id: int = -1

    running = True
    while running:
        frame, frame_id = stream.get_latest_with_id()
        if frame is not None:
            state.frame_h, state.frame_w = frame.shape[:2]
            if frame_id != _prev_frame_id:
                _adb_frame_count += 1
                _prev_frame_id = frame_id

        # Recompute ADB FPS every second
        _now = time.monotonic()
        _elapsed = _now - _adb_fps_window_start
        if _elapsed >= 1.0:
            _adb_fps = _adb_frame_count / _elapsed
            _adb_frame_count = 0
            _adb_fps_window_start = _now

        # ── Events ───────────────────────────────────────────────────────
        for event in pygame.event.get():
            if not dispatch_event(
                event,
                state=state,
                device=device,
                recognizer=recognizer,
                frame=frame,
            ):
                running = False

        # ── Pull latest analysis from the worker thread ──────────────────
        (
            analysis_frame_id,
            snap_board_grid,
            snap_queue,
            snap_confidences,
            suggestion,
        ) = analyzer.snapshot()
        state.board.grid  = snap_board_grid
        queue             = snap_queue
        queue_confidences = snap_confidences
        analysis_changed  = analysis_frame_id != last_analysis_frame_id
        if analysis_changed:
            last_analysis_frame_id = analysis_frame_id

        # ── Auto-play execution ──────────────────────────────────────────
        maybe_execute_auto_swipe(
            device=device,
            state=state,
            analyzer=analyzer,
            suggestion=suggestion,
            queue=queue,
            queue_confidences=queue_confidences,
            analysis_changed=analysis_changed,
            analysis_frame_id=analysis_frame_id,
        )

        # While a servo placement is in flight, freeze the queue CNN's
        # output (suggestion / queue / confidences) to the snapshot taken
        # at dispatch time.  The board grid still updates live so the
        # recon panel shows the held piece's solid render drifting into
        # place.
        if state.servo_active:
            suggestion        = state.frozen_suggestion
            queue             = state.frozen_queue
            queue_confidences = state.frozen_confidences

        # ── Render ───────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

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
        state.scale, state.blit_x, state.blit_y = draw_phone_panel(
            screen,
            frame=frame,
            rect=PHONE_RECT,
            error_msg=stream.last_error,
            font=font,
            small_font=small_font,
            cached_surface=cached,
        )

        if frame is not None:
            if state.cfg.grid is not None:
                draw_grid_overlay(screen, state.cfg.grid, state.scale, state.blit_x, state.blit_y)
            if state.cfg.queue is not None:
                draw_queue_overlay(
                    screen, state.cfg.queue, state.scale, state.blit_x, state.blit_y, small_font,
                    chosen_slot=suggestion.slot if suggestion is not None else None,
                )
            if suggestion is not None and state.cfg.grid is not None and state.cfg.grid.is_valid():
                draw_suggestion_on_phone(
                    screen, state.cfg.grid, suggestion,
                    state.scale, state.blit_x, state.blit_y,
                )
            if state.servo_debug_view:
                draw_servo_error_on_phone(
                    screen,
                    target_xy=state.servo_target_px,
                    measured_xy=state.servo_measured_px,
                    target_cells=state.servo_target_cells,
                    measured_cells=state.servo_measured_cells,
                    scale=state.scale,
                    blit_x=state.blit_x,
                    blit_y=state.blit_y,
                    small_font=small_font,
                )
            # Auto-play swipe arrow — visualise the last issued drag
            if state.last_swipe is not None:
                src_xy, dst_xy, t_start, dur_ms = state.last_swipe
                age_ms = pygame.time.get_ticks() - t_start
                draw_swipe_arrow_on_phone(
                    screen,
                    src_xy, dst_xy,
                    age_ms=age_ms,
                    scale=state.scale,
                    blit_x=state.blit_x,
                    blit_y=state.blit_y,
                    duration_ms=dur_ms,
                    small_font=small_font,
                )

        if state.drag_start is not None and state.drag_cur is not None:
            draw_drag_preview(screen, state.drag_start, state.drag_cur)

        # Recon panel — rebuilt off-screen only when the underlying state changes.
        new_recon_key = (
            state.board.grid.tobytes(),
            state.board.grid.shape,
            tuple(p.piece_id if p is not None else -1 for p in queue),
            tuple(round(c, 3) for c in queue_confidences),
            (
                (suggestion.slot, suggestion.row,
                 suggestion.col, suggestion.piece.piece_id)
                if suggestion is not None else None
            ),
            (
                # Round the score so the cache key doesn't churn on
                # micro-fluctuations between frames.
                (state.servo_detection[0], state.servo_detection[1],
                 state.servo_detection[2], state.servo_detection[3],
                 round(state.servo_detection[4], 2))
                if state.servo_detection is not None else None
            ),
            state.servo_debug_view,
            # When the debug view is on, invalidate the cache per-frame
            # while a mask is available so the live motion view animates.
            (id(state.servo_debug_mask)
             if state.servo_debug_view and state.servo_debug_mask is not None
             else None),
            (id(state.servo_debug_mask_rolling)
             if state.servo_debug_view and state.servo_debug_mask_rolling is not None
             else None),
        )
        if new_recon_key != recon_cache_key or recon_cache_surf is None:
            recon_cache_surf = pygame.Surface(RECON_RECT.size, pygame.SRCALPHA)
            draw_recon_panel(
                recon_cache_surf,
                rect=pygame.Rect(0, 0, RECON_RECT.width, RECON_RECT.height),
                board=state.board,
                queue=queue,
                font=font,
                small_font=small_font,
                suggestion=suggestion,
                queue_confidences=queue_confidences,
                detection=state.servo_detection,
                debug_mask=state.servo_debug_mask,
                debug_mask_rolling=state.servo_debug_mask_rolling,
                debug_view=state.servo_debug_view,
            )
            recon_cache_key = new_recon_key
        screen.blit(recon_cache_surf, RECON_RECT.topleft)

        draw_status_bar(
            screen,
            fps=clock.get_fps(),
            has_device=frame is not None,
            rect=STATUS_RECT,
            small_font=small_font,
            hint="",
            adb_fps=_adb_fps,
        )

        state.control_rects = draw_controls_panel(
            screen,
            CONTROLS_RECT,
            small_font,
            calib_mode=state.calib_mode,
            auto_enabled=state.auto_enabled,
            device_supports_input=getattr(device, "supports_input", False),
            servo_debug_view=state.servo_debug_view,
        )

        pygame.display.flip()
        clock.tick(_TARGET_FPS)

    analyzer.stop()
    stream.stop()
    pygame.quit()
