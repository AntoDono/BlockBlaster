"""Keyboard and chip-click handlers for the assist pygame app."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import pygame

from blockblaster.assist.ui.state import AppState

_SCROLL_LINES_PER_TICK = 3

_SCREENSHOTS_DIR = Path(__file__).resolve().parents[3] / "screenshots"


def dispatch_event(event: pygame.event.Event, *, state: AppState) -> bool:
    """Route one pygame event. Returns False when the user requested quit."""
    if event.type == pygame.MOUSEWHEEL:
        if state.log_rect.collidepoint(pygame.mouse.get_pos()):
            state.log_scroll = max(0, state.log_scroll + event.y * _SCROLL_LINES_PER_TICK)
            return True
    if event.type == pygame.QUIT:
        return False
    if event.type == pygame.KEYDOWN:
        return _handle_keydown(event, state)
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        action = _hit_test_chips(event.pos, state)
        if action == "quit":
            return False
        elif action == "screenshot":
            _save_screenshot()
        elif action == "autoplay":
            state.autoplay_on = not state.autoplay_on
        elif action == "debug":
            state.show_debug = not state.show_debug
        elif action == "editboard":
            state.edit_board = not state.edit_board
        elif action == "recalibrate":
            state.reset_analysis_request = True
        elif action is None and state.edit_board:
            f = _screen_to_frame(event.pos, state)
            if f is not None:
                state.drag_start_frame = f
                state.drag_cur_frame = f
    if event.type == pygame.MOUSEMOTION and state.drag_start_frame is not None:
        f = _screen_to_frame(event.pos, state)
        if f is not None:
            state.drag_cur_frame = f
    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        if state.drag_start_frame is not None:
            _finish_board_drag(event.pos, state)
    return True


def _finish_board_drag(pos: tuple[int, int], state: AppState) -> None:
    """Commit the dragged box as the manual board override (frame px)."""
    end = _screen_to_frame(pos, state) or state.drag_cur_frame
    start = state.drag_start_frame
    state.drag_start_frame = None
    state.drag_cur_frame = None
    if start is None or end is None:
        return
    x = min(start[0], end[0])
    y = min(start[1], end[1])
    w = abs(end[0] - start[0])
    h = abs(end[1] - start[1])
    if w > 10 and h > 10:
        state.board_override = (x, y, w, h)
        print(f"[edit] board override set: {(x, y, w, h)}")


def _screen_to_frame(
    pos: tuple[int, int], state: AppState
) -> Optional[tuple[int, int]]:
    """Map a phone-panel screen point to frame px, or None if outside the panel."""
    scale, bx, by = state.phone_map
    if scale <= 0:
        return None
    fx = (pos[0] - bx) / scale
    fy = (pos[1] - by) / scale
    if fx < 0 or fy < 0 or fx > state.frame_w or fy > state.frame_h:
        return None
    return int(fx), int(fy)


def _handle_keydown(event: pygame.event.Event, state: AppState) -> bool:
    if event.key in (pygame.K_q, pygame.K_ESCAPE):
        return False
    if event.key == pygame.K_s:
        _save_screenshot()
    if event.key == pygame.K_a:
        state.autoplay_on = not state.autoplay_on
    if event.key == pygame.K_d:
        state.show_debug = not state.show_debug
    if event.key == pygame.K_e:
        state.edit_board = not state.edit_board
    if event.key == pygame.K_r:
        state.reset_analysis_request = True
    return True


def _hit_test_chips(pos: tuple[int, int], state: AppState) -> Optional[str]:
    for action, chip_rect in state.control_rects.items():
        if chip_rect.collidepoint(pos):
            return action
    return None


def _save_screenshot() -> None:
    surface = pygame.display.get_surface()
    if surface is None:
        print("[screenshot] no pygame surface yet — ignored.")
        return
    try:
        _SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = _SCREENSHOTS_DIR / f"assist_{ts}.png"
        pygame.image.save(surface, str(path))
        print(f"[screenshot] saved → {path}")
    except Exception as exc:
        print(f"[screenshot] failed: {exc}")
