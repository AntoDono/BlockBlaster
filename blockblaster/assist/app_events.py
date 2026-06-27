"""Keyboard and chip-click handlers for the assist pygame app."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import pygame

from blockblaster.assist.app_state import AppState

_SCREENSHOTS_DIR = Path(__file__).resolve().parents[2] / "screenshots"


def dispatch_event(event: pygame.event.Event, *, state: AppState) -> bool:
    """Route one pygame event. Returns False when the user requested quit."""
    if event.type == pygame.QUIT:
        return False
    if event.type == pygame.KEYDOWN:
        return _handle_keydown(event)
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        action = _hit_test_chips(event.pos, state)
        if action == "quit":
            return False
        if action == "screenshot":
            _save_screenshot()
    return True


def _handle_keydown(event: pygame.event.Event) -> bool:
    if event.key in (pygame.K_q, pygame.K_ESCAPE):
        return False
    if event.key == pygame.K_s:
        _save_screenshot()
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
