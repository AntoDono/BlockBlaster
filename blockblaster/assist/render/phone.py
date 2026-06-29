"""Phone-panel and status-bar rendering for the assist GUI."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pygame

from blockblaster.assist.vision.detection import Element, annotate

PANEL_BG       = (20, 20, 30)
PANEL_BORDER   = (55, 55, 75)
TEXT_COLOR     = (220, 220, 235)
DIM_TEXT       = (110, 110, 130)
STATUS_BG      = (10, 10, 16)
DEVICE_OK_COL  = (60, 220, 100)
DEVICE_ERR_COL = (220, 70, 70)
LABEL_COL      = (160, 160, 185)
SUGGEST_FILL   = (80, 240, 120)
SUGGEST_BORDER = (40, 200, 80)
SUGGEST_FILL_A = 170
SUGGEST_GOLD   = (255, 200, 40)  # shared accent for outlines and edit overlay


def panel_content_rect(panel_rect: pygame.Rect) -> pygame.Rect:
    """Inner content area of a panel (under the title strip, padded)."""
    return pygame.Rect(
        panel_rect.x + 4, panel_rect.y + 30,
        panel_rect.width - 8, panel_rect.height - 38,
    )


def bgr_to_surface(
    frame_bgr: np.ndarray,
    target_rect: pygame.Rect,
) -> tuple[pygame.Surface, float, int, int]:
    """Letterbox ``frame_bgr`` into ``target_rect``; returns ``(surf, scale, x, y)``."""
    h, w   = frame_bgr.shape[:2]
    scale  = min(target_rect.width / w, target_rect.height / h)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    rgb     = np.ascontiguousarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    surf    = pygame.image.frombuffer(rgb.tobytes(), (new_w, new_h), "RGB")
    blit_x  = target_rect.x + (target_rect.width  - new_w) // 2
    blit_y  = target_rect.y + (target_rect.height - new_h) // 2
    return surf, scale, blit_x, blit_y


def draw_phone_panel(
    screen: pygame.Surface,
    frame: Optional[np.ndarray],
    elements: list[Element],
    rect: pygame.Rect,
    error_msg: Optional[str],
    small_font: pygame.font.Font,
) -> None:
    """Draw the left phone-screen panel with detection overlays baked in."""
    pygame.draw.rect(screen, PANEL_BG,     rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER, rect, width=2, border_radius=10)

    lbl = small_font.render("PHONE SCREEN", True, LABEL_COL)
    screen.blit(lbl, (rect.x + 10, rect.y + 8))

    content = panel_content_rect(rect)
    if frame is not None:
        annotated = annotate(frame, elements) if elements else frame
        surf, _, bx, by = bgr_to_surface(annotated, content)
        screen.blit(surf, (bx, by))
        return

    msg   = error_msg or "Waiting for device…"
    lines = _wrap(msg, small_font, content.width - 20)
    total_h = len(lines) * (small_font.get_height() + 4)
    cy = content.centery - total_h // 2
    for line in lines:
        s = small_font.render(line, True, DIM_TEXT)
        screen.blit(s, (content.centerx - s.get_width() // 2, cy))
        cy += small_font.get_height() + 4


def draw_status_bar(
    screen: pygame.Surface,
    fps: float,
    has_device: bool,
    rect: pygame.Rect,
    small_font: pygame.font.Font,
    hint: str = "",
    adb_fps: float = 0.0,
    device_detail: str = "",
) -> None:
    pygame.draw.rect(screen, STATUS_BG, rect)

    device_col = DEVICE_OK_COL if has_device else DEVICE_ERR_COL
    if has_device:
        extra = f" — {device_detail}" if device_detail else ""
        device_txt = f"Device: connected{extra}"
    else:
        device_txt = "Device: not connected"
    d = small_font.render(device_txt, True, device_col)
    max_w = rect.width - 280
    if d.get_width() > max_w:
        device_txt = _truncate(device_txt, small_font, max_w)
        d = small_font.render(device_txt, True, device_col)
    screen.blit(d, (rect.x + 12, rect.y + (rect.height - d.get_height()) // 2))

    app_fps_s = small_font.render(f"App {fps:.0f} fps", True, DIM_TEXT)
    adb_col   = DEVICE_OK_COL if adb_fps >= 5 else DEVICE_ERR_COL
    adb_fps_s = small_font.render(f"ADB {adb_fps:.1f} fps", True, adb_col)

    app_x = rect.right - app_fps_s.get_width() - 12
    adb_x = app_x - adb_fps_s.get_width() - 16
    mid_y = rect.y + (rect.height - app_fps_s.get_height()) // 2
    screen.blit(app_fps_s, (app_x, mid_y))
    screen.blit(adb_fps_s, (adb_x, mid_y))

    if hint:
        hint_s = small_font.render(hint, True, DIM_TEXT)
        screen.blit(hint_s, (rect.centerx - hint_s.get_width() // 2,
                             rect.y + (rect.height - hint_s.get_height()) // 2))


def _truncate(text: str, font: pygame.font.Font, max_w: int) -> str:
    if font.size(text)[0] <= max_w:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.size(text[:mid] + ell)[0] <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ell


def _wrap(text: str, font: pygame.font.Font, max_w: int) -> list[str]:
    words, lines, current = text.split(), [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_w:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]
