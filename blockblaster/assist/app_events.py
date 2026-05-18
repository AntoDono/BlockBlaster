"""Keyboard and mouse event handlers for the assist pygame app.

Extracted from :mod:`app` so the main run loop stays narrow.  Handlers mutate
the shared :class:`AppState` and return ``False`` from :func:`dispatch_event`
to signal that the app should quit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame

from blockblaster.assist.app_state import MODE_GRID, MODE_PIECES, AppState
from blockblaster.assist.calibration import CalibrationBox
from blockblaster.assist.layout import PHONE_RECT
from blockblaster.assist.piece_recognizer import PieceRecognizer
from blockblaster.control.device import Device
from blockblaster.game.board import Board


def dispatch_event(
    event: pygame.event.Event,
    *,
    state: AppState,
    device: Device,
    recognizer: PieceRecognizer,
    frame: Optional[np.ndarray],
) -> bool:
    """Route one pygame event to the appropriate handler.

    Returns ``False`` when the user requested quit; ``True`` otherwise.
    """
    if event.type == pygame.QUIT:
        return False
    if event.type == pygame.KEYDOWN:
        return _handle_keydown(event, state=state, device=device, recognizer=recognizer, frame=frame)
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        action = _hit_test_chips(event.pos, state)
        if action is not None:
            return _dispatch_chip(action, state=state, device=device,
                                  recognizer=recognizer, frame=frame)
        _handle_mouse_down(event, state=state, frame=frame)
    elif event.type == pygame.MOUSEMOTION:
        if state.drag_start is not None:
            state.drag_cur = event.pos
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        _handle_mouse_up(state=state, frame=frame)
    return True


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------

def _handle_keydown(
    event: pygame.event.Event,
    *,
    state: AppState,
    device: Device,
    recognizer: PieceRecognizer,
    frame: Optional[np.ndarray],
) -> bool:
    """Returns False if the keybinding requests quit."""
    if event.key in (pygame.K_q, pygame.K_ESCAPE):
        return False

    if event.key == pygame.K_TAB:
        state.calib_mode = MODE_PIECES if state.calib_mode == MODE_GRID else MODE_GRID

    elif event.key == pygame.K_r:
        if state.calib_mode == MODE_GRID:
            state.cfg.grid = None
            state.board = Board()
        else:
            state.cfg.queue = None
        state.cfg.save(platform=state.platform)

    elif event.key == pygame.K_d:
        if frame is not None and state.cfg.queue is not None and state.cfg.queue.is_valid():
            out = recognizer.save_debug(frame, state.cfg.queue)
            print(f"Saved queue debug images to: {out.resolve()}")
        else:
            print("Debug skipped: need a frame and a calibrated queue box.")

    elif event.key == pygame.K_a:
        _toggle_auto_play(state, device)

    elif event.key == pygame.K_t:
        _test_swipe(state, device)

    elif event.key == pygame.K_h:
        _test_hold(state, device)

    return True


def _test_swipe(state: AppState, device: Device) -> None:
    """Issue a known-safe diagnostic swipe in the middle of the grid.

    Bypasses calibration and auto-play.  If THIS lands as a drag in Block
    Blast → the input pipeline is fine and the bug is in the queue/grid
    calibration.  If this also opens the system menu → the device is
    rejecting `input swipe` for some other reason (speed, gesture nav,
    accessibility, etc.).
    """
    if not getattr(device, "supports_input", False):
        print("[test_swipe] device does not support input.")
        return
    try:
        w, h = device.screen_size()
    except Exception as exc:
        print(f"[test_swipe] screen_size failed: {exc}")
        return
    # Middle 60% of the screen, swipe a short vertical segment well clear
    # of the bottom 25% (gesture-nav zone) and the top 10% (status bar).
    cx       = w // 2
    y_start  = int(h * 0.55)
    y_end    = int(h * 0.40)
    duration = 900
    print(
        f"[test_swipe] device={w}x{h} → swipe ({cx},{y_start}) → "
        f"({cx},{y_end}) duration={duration}ms"
    )
    try:
        device.swipe(cx, y_start, cx, y_end, duration_ms=duration)
        # Show it on the GUI too so user sees the same line drawn.
        state.last_swipe = ((cx, y_start), (cx, y_end), pygame.time.get_ticks(), duration)
    except Exception as exc:
        print(f"[test_swipe] swipe failed: {exc}")


def _test_hold(state: AppState, device: Device) -> None:
    """Press-and-hold on queue slot 1 for 2s so you can see if Block Blast
    selects the piece (it should visually pop / highlight under your "finger").

    Uses ``input swipe x y x y 2000`` which Android implements as a stationary
    long-press.  If the piece highlights → the touch is reaching the game and
    the issue is the *drag motion*.  If nothing happens → the synthetic DOWN
    isn't being registered at that location at all.
    """
    if not getattr(device, "supports_input", False):
        print("[test_hold] device does not support input.")
        return
    if state.cfg.queue is None:
        print("[test_hold] queue not calibrated.")
        return
    from blockblaster.control.coords import slot_center_px
    sx, sy = slot_center_px(state.cfg.queue, 0)
    duration = 2000
    print(f"[test_hold] hold at ({sx},{sy}) for {duration}ms — watch the piece")
    try:
        device.swipe(sx, sy, sx, sy, duration_ms=duration)
        state.last_swipe = ((sx, sy), (sx, sy), pygame.time.get_ticks(), duration)
    except Exception as exc:
        print(f"[test_hold] failed: {exc}")


def _toggle_auto_play(state: AppState, device: Device) -> None:
    if not getattr(device, "supports_input", False):
        print("[assist] device does not support input — cannot enable auto-play.")
        return
    state.auto_enabled = not state.auto_enabled
    print(f"[assist] auto-play {'ON' if state.auto_enabled else 'OFF'}")


# ---------------------------------------------------------------------------
# Chip hit-testing and dispatch
# ---------------------------------------------------------------------------

def _hit_test_chips(pos: tuple[int, int], state: AppState) -> Optional[str]:
    """Return the action name if ``pos`` is inside a chip, else ``None``."""
    for action, chip_rect in state.control_rects.items():
        if chip_rect.collidepoint(pos):
            return action
    return None


def _dispatch_chip(
    action: str,
    *,
    state: AppState,
    device: Device,
    recognizer: PieceRecognizer,
    frame: Optional[np.ndarray],
) -> bool:
    """Handle a chip click.  Returns False only for the 'quit' action."""
    if action == "quit":
        return False
    if action == "auto":
        _toggle_auto_play(state, device)
    elif action == "mode":
        state.calib_mode = MODE_PIECES if state.calib_mode == MODE_GRID else MODE_GRID
    elif action == "clear":
        if state.calib_mode == MODE_GRID:
            state.cfg.grid = None
            state.board = Board()
        else:
            state.cfg.queue = None
        state.cfg.save(platform=state.platform)
    elif action == "debug":
        if frame is not None and state.cfg.queue is not None and state.cfg.queue.is_valid():
            out = recognizer.save_debug(frame, state.cfg.queue)
            print(f"Saved queue debug images to: {out.resolve()}")
        else:
            print("Debug skipped: need a frame and a calibrated queue box.")
    return True


# ---------------------------------------------------------------------------
# Mouse
# ---------------------------------------------------------------------------

def _handle_mouse_down(
    event: pygame.event.Event,
    *,
    state: AppState,
    frame: Optional[np.ndarray],
) -> None:
    mx, my = event.pos
    if PHONE_RECT.collidepoint(mx, my) and frame is not None:
        state.drag_start = (mx, my)
        state.drag_cur   = (mx, my)


def _handle_mouse_up(*, state: AppState, frame: Optional[np.ndarray]) -> None:
    if state.drag_start is None or state.drag_cur is None or frame is None:
        state.drag_start = None
        state.drag_cur   = None
        return

    new_box = CalibrationBox.from_screen(
        state.drag_start[0], state.drag_start[1],
        state.drag_cur[0],   state.drag_cur[1],
        state.scale, state.blit_x, state.blit_y,
        state.frame_w, state.frame_h,
    )
    if new_box.is_valid():
        if state.calib_mode == MODE_GRID:
            state.cfg.grid = new_box
        else:
            state.cfg.queue = new_box
        state.cfg.save(platform=state.platform)

    state.drag_start = None
    state.drag_cur   = None
