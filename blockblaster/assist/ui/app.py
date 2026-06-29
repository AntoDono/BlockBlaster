"""Pygame application loop for the assist side-by-side viewer."""

from __future__ import annotations

import threading
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
_AUTO_POST_PLACE_S = 1.5


def _phone_content() -> pygame.Rect:
    """Inner content rect of the phone panel (matches draw_phone_panel)."""
    return pygame.Rect(PHONE_RECT.x + 4, PHONE_RECT.y + 30,
                       PHONE_RECT.width - 8, PHONE_RECT.height - 38)


def _phone_map(frame_w: int, frame_h: int) -> tuple[float, int, int]:
    """(scale, blit_x, blit_y) mapping frame px → phone-panel screen px."""
    if frame_w <= 0 or frame_h <= 0:
        return (1.0, 0, 0)
    c = _phone_content()
    scale = min(c.width / frame_w, c.height / frame_h)
    bx = c.x + (c.width - int(frame_w * scale)) // 2
    by = c.y + (c.height - int(frame_h * scale)) // 2
    return (scale, bx, by)


def _draw_board_edit(
    screen: pygame.Surface, state: AppState, font: pygame.font.Font
) -> None:
    """Draw the manual board override box / in-progress drag on the phone panel."""
    scale, bx, by = state.phone_map

    def to_screen(fx: int, fy: int) -> tuple[int, int]:
        return (bx + int(fx * scale), by + int(fy * scale))

    if state.drag_start_frame is not None and state.drag_cur_frame is not None:
        x0, y0 = to_screen(*state.drag_start_frame)
        x1, y1 = to_screen(*state.drag_cur_frame)
        rect = pygame.Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        pygame.draw.rect(screen, (255, 200, 40), rect, width=2)
    elif state.board_override is not None:
        ox, oy, ow, oh = state.board_override
        tl = to_screen(ox, oy)
        rect = pygame.Rect(tl[0], tl[1], int(ow * scale), int(oh * scale))
        pygame.draw.rect(screen, (255, 200, 40), rect, width=2)

    if state.edit_board:
        msg = font.render("EDIT BOARD: drag a box on the phone screen",
                          True, (255, 200, 40))
        screen.blit(msg, (PHONE_RECT.x + 10, PHONE_RECT.y + PHONE_RECT.height - 24))


def _maybe_dispatch_servo(device: Device, state: AppState, snap) -> None:
    """Run one closed-loop servo placement of the current suggestion (key 'a').

    Grabs the tray piece at its detected bbox centre and servos it onto the
    suggested board cells on a worker thread so the GUI keeps rendering.
    """
    from blockblaster.control.servo import place

    sug = snap.suggestion
    if sug is None or snap.board_bbox is None:
        return
    if not (0 <= sug.slot < len(snap.pieces)):
        return

    px, py, pw, ph = snap.pieces[sug.slot].bbox
    grab_px = (int(px + pw / 2), int(py + ph / 2))
    grid_bbox = snap.board_bbox
    frame_w, frame_h = state.frame_w, state.frame_h

    def _on_debug(dbg) -> None:
        state.servo_debug = dbg

    def _worker() -> None:
        try:
            ok = place(
                device=device,
                grid_bbox=grid_bbox,
                grab_px=grab_px,
                suggestion=sug,
                frame_w=frame_w,
                frame_h=frame_h,
                on_debug=_on_debug,
            )
            print(f"[auto] servo: {'ok' if ok else 'FAIL'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[auto] servo crashed: {exc!r}")
        finally:
            state.auto_next_after = time.monotonic() + _AUTO_POST_PLACE_S
            state.servo_busy = False
            state.servo_debug = None

    state.servo_busy = True
    threading.Thread(target=_worker, name="auto-servo", daemon=True).start()


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

        if state.recalibrate_request:
            state.recalibrate_request = False
            analyzer.recalibrate()
            diff_tracker.clear_suggestion()
            state.board_override = None  # revert to auto-detection

        analyzer.set_board_override(state.board_override)
        state.phone_map = _phone_map(state.frame_w, state.frame_h)

        if state.autoplay_on and not state.servo_busy and now >= state.auto_next_after:
            _maybe_dispatch_servo(device, state, snap)

        screen.fill(BG_COLOR)

        draw_phone_panel(
            screen,
            frame=frame,
            elements=snap.elements,
            rect=PHONE_RECT,
            error_msg=device.last_error,
            small_font=small_font,
        )
        _draw_board_edit(screen, state, small_font)

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
            servo_debug=state.servo_debug if state.show_debug else None,
        )

        draw_status_bar(
            screen,
            fps=clock.get_fps(),
            has_device=frame is not None,
            rect=STATUS_RECT,
            small_font=small_font,
            adb_fps=adb_fps,
        )

        state.control_rects = draw_controls_panel(
            screen, CONTROLS_RECT, small_font,
            autoplay_on=state.autoplay_on,
            servo_busy=state.servo_busy,
            show_debug=state.show_debug,
            edit_board=state.edit_board,
        )

        pygame.display.flip()
        clock.tick(_TARGET_FPS)

    analyzer.stop()
    device.stop()
    pygame.quit()
