"""Pygame application loop for the assist side-by-side viewer."""

from __future__ import annotations

import time
from typing import Optional

import pygame

from blockblaster.assist.advisor import Advisor
from blockblaster.assist.vision.analyzer import AnalysisWorker
from blockblaster.assist.ui.events import dispatch_event
from blockblaster.assist.ui.overlay import draw_controls_panel
from blockblaster.assist.ui.state import AppState
from blockblaster.assist.ui.layout import (
    BG_COLOR,
    CNN_DEBUG_RECT,
    CONTROLS_RECT,
    FRAME_DIFF_RECT,
    PHONE_RECT,
    RECON_RECT,
    STATUS_RECT,
    make_window,
)
from blockblaster.assist.vision.frame_diff import FrameDiffTracker
from blockblaster.assist.vision.piece_recognizer import PieceRecognizer
from blockblaster.assist.render import (
    draw_cnn_debug_panel,
    draw_frame_diff_panel,
    draw_phone_panel,
    draw_recon_panel,
    draw_status_bar,
)
from blockblaster.control.device import Device

_TARGET_FPS = 60


def run(
    device: Optional[Device] = None,
    platform: Optional[str] = None,
) -> None:
    """Launch the assist viewer (left: annotated phone, right: reconstruction)."""
    pygame.init()
    pygame.font.init()

    font       = pygame.font.SysFont("monospace", 20, bold=True)
    small_font = pygame.font.SysFont("monospace", 15)

    screen = make_window()
    clock  = pygame.time.Clock()

    recognizer = PieceRecognizer()
    advisor    = Advisor()
    if advisor.last_error:
        print(f"[assist] advisor disabled: {advisor.last_error}")

    if device is None:
        from blockblaster.control.ios_readonly import IosReadOnlyDevice
        device = IosReadOnlyDevice()
    device.start()

    diff_tracker = FrameDiffTracker()
    analyzer = AnalysisWorker(
        device=device,
        recognizer=recognizer,
        advisor=advisor,
        diff_tracker=diff_tracker,
    )
    analyzer.start()

    state = AppState(platform=platform)

    adb_window_start = time.monotonic()
    adb_count        = 0
    adb_fps          = 0.0
    prev_frame_id    = -1

    running = True
    while running:
        frame, frame_id = device.get_latest_with_id()
        now = time.monotonic()
        if frame is not None:
            state.frame_h, state.frame_w = frame.shape[:2]
            if frame_id != prev_frame_id:
                adb_count    += 1
                prev_frame_id = frame_id
                diff_tracker.observe(frame, now)

        elapsed = now - adb_window_start
        if elapsed >= 1.0:
            adb_fps          = adb_count / elapsed
            adb_count        = 0
            adb_window_start = now

        for event in pygame.event.get():
            if not dispatch_event(event, state=state):
                running = False

        snap = analyzer.snapshot()
        diff_tracker.set_suggestion(snap.suggestion, snap.board_bbox)

        screen.fill(BG_COLOR)

        draw_phone_panel(
            screen,
            frame=frame,
            elements=snap.elements,
            rect=PHONE_RECT,
            error_msg=device.last_error,
            small_font=small_font,
        )

        draw_recon_panel(
            screen,
            rect=RECON_RECT,
            snap=snap,
            frame_w=state.frame_w,
            frame_h=state.frame_h,
            small_font=small_font,
        )

        draw_cnn_debug_panel(
            screen,
            rect=CNN_DEBUG_RECT,
            snap=snap,
            small_font=small_font,
        )

        draw_frame_diff_panel(
            screen,
            rect=FRAME_DIFF_RECT,
            tracker=diff_tracker,
            now=now,
            small_font=small_font,
        )

        draw_status_bar(
            screen,
            fps=clock.get_fps(),
            has_device=frame is not None,
            rect=STATUS_RECT,
            small_font=small_font,
            adb_fps=adb_fps,
        )

        state.control_rects = draw_controls_panel(screen, CONTROLS_RECT, small_font)

        pygame.display.flip()
        clock.tick(_TARGET_FPS)

    analyzer.stop()
    device.stop()
    pygame.quit()
