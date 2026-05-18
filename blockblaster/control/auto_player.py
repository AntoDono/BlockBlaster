"""Headless Android auto-player.

Capture → scan → recognise → advise → swipe → wait → repeat.

Usage::

    uv run play.py --platform android [--display] [--serial <serial>]

or directly::

    uv run python -m blockblaster.control.auto_player [--display] [--serial <serial>]
"""

from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from blockblaster.control.device import Device

CONF_THRESHOLD    = 0.65   # skip a frame if any slot confidence is below this
POST_PLACE_MS     = 600    # wait after each swipe (animation + queue refresh)
CHANGE_TIMEOUT_MS = 2000   # give up waiting for a frame change after this
DISPLAY_SCALE     = 0.45   # pygame preview window scale factor


def run(
    serial: Optional[str] = None,
    display: bool = False,
) -> None:
    """Launch the auto-play loop.

    Parameters
    ----------
    serial:
        ADB device serial.  Auto-detected when ``None``.
    display:
        When ``True``, open a pygame window showing the live frame and the
        planned drag annotated on top.
    """
    from blockblaster.assist.advisor import Advisor
    from blockblaster.assist.calibration import CalibrationConfig
    from blockblaster.assist.piece_recognizer import PieceRecognizer
    from blockblaster.assist.scanner import scan_board
    from blockblaster.control.coords import piece_anchor_px, slot_center_px
    from blockblaster.control.device import make_device
    from blockblaster.control.visual_servo import _GRAB_Y_NUDGE_PX, place_with_servo
    from blockblaster.game.board import Board

    cfg = CalibrationConfig.load(platform="android")
    if cfg.grid is None or not cfg.grid.is_valid():
        print(
            "[auto_player] Grid calibration box not set for android platform.\n"
            "Run `play.py --platform android --mode assist` to calibrate, then retry."
        )
        return
    if cfg.queue is None or not cfg.queue.is_valid():
        print(
            "[auto_player] Queue calibration box not set for android platform.\n"
            "Run `play.py --platform android --mode assist` to calibrate, then retry."
        )
        return

    device = make_device("android", serial=serial)
    device.start()
    print(f"[auto_player] Connected to {getattr(device, '_serial', '?')}")

    recognizer = PieceRecognizer()
    advisor    = Advisor()
    if advisor.last_error:
        print(f"[auto_player] Advisor warning: {advisor.last_error}")

    screen_surf = None
    if display:
        import pygame
        pygame.init()
        pygame.font.init()
        w, h = device.screen_size()
        win_w = max(1, int(w * DISPLAY_SCALE))
        win_h = max(1, int(h * DISPLAY_SCALE))
        screen_surf = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("BlockBlaster auto-player")
        clock = pygame.time.Clock()

    last_frame_id = -1
    consecutive_no_move = 0

    try:
        while True:
            if display:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key in (
                        pygame.K_q, pygame.K_ESCAPE
                    ):
                        return

            frame, frame_id = device.get_latest_with_id()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.05)
                continue

            last_frame_id = frame_id

            # ── Scan board + queue ─────────────────────────────────────────
            board_obj = Board()
            board_obj.grid = scan_board(frame, cfg.grid)

            results  = recognizer.recognize_queue_with_confidence(frame, cfg.queue)
            queue    = [p for p, _ in results]
            confs    = [c for _, c in results]
            pieces   = [p for p in queue if p is not None]

            if not pieces:
                print("[auto_player] Queue appears empty — game over or scan failed.")
                consecutive_no_move += 1
                if consecutive_no_move >= 5:
                    print("[auto_player] Too many empty queues in a row — stopping.")
                    break
                time.sleep(0.3)
                continue

            low_conf = [f"slot{i+1}={c:.2f}" for i, c in enumerate(confs) if c < CONF_THRESHOLD]
            if low_conf:
                print(f"[auto_player] Low confidence {low_conf}, waiting for better frame…")
                time.sleep(0.1)
                continue

            # ── Get suggestion ────────────────────────────────────────────
            suggestion = advisor.suggest(board_obj.grid, queue)
            if suggestion is None:
                print("[auto_player] No legal move — game over.")
                break

            consecutive_no_move = 0

            # Annotate planned drag for the preview window (frame px).
            slot_cx, slot_cy = slot_center_px(cfg.queue, suggestion.slot)
            src = (slot_cx, slot_cy - _GRAB_Y_NUDGE_PX)
            dst = piece_anchor_px(
                cfg.grid, suggestion.piece, suggestion.row, suggestion.col,
            )

            print(
                f"[auto_player] {suggestion.piece.name} slot={suggestion.slot+1} "
                f"→ row={suggestion.row+1} col={suggestion.col+1}  "
                f"  servo {src} → {dst}  conf=[{', '.join(f'{c:.2f}' for c in confs)}]"
            )

            if display and screen_surf is not None:
                import pygame
                _render_display(screen_surf, frame, src, dst, suggestion, confs, DISPLAY_SCALE)
                pygame.display.flip()
                clock.tick(30)

            fh, fw = frame.shape[:2]
            result = place_with_servo(
                device=device,
                cfg=cfg,
                suggestion=suggestion,
                frame_w=fw,
                frame_h=fh,
            )
            print(
                f"[auto_player] servo: {'ok' if result.success else 'FAIL'} "
                f"({result.reason}, {result.iters} iters)"
            )

            _wait_for_change(device, last_frame_id, CHANGE_TIMEOUT_MS)
            time.sleep(POST_PLACE_MS / 1000)

    finally:
        device.stop()
        if display:
            import pygame
            pygame.quit()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wait_for_change(device: "Device", old_id: int, timeout_ms: int) -> None:
    """Block until the device's frame_id advances past *old_id*."""
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        _, fid = device.get_latest_with_id()
        if fid != old_id:
            return
        time.sleep(0.04)


def _render_display(
    screen: "pygame.Surface",
    frame: np.ndarray,
    src: tuple[int, int],
    dst: tuple[int, int],
    suggestion: "Suggestion",
    confs: list[float],
    scale: float,
) -> None:
    """Draw the annotated frame into the pygame window."""
    import pygame

    h, w    = frame.shape[:2]
    preview = cv2.resize(frame, (int(w * scale), int(h * scale)))
    preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    surf    = pygame.image.frombuffer(
        np.ascontiguousarray(preview).tobytes(),
        (preview.shape[1], preview.shape[0]), "RGB",
    )
    screen.blit(surf, (0, 0))

    # Drag arrow
    s = (int(src[0] * scale), int(src[1] * scale))
    d = (int(dst[0] * scale), int(dst[1] * scale))
    pygame.draw.circle(screen, (80, 240, 120), s, 8)
    pygame.draw.line(screen,   (80, 240, 120), s, d, 3)
    pygame.draw.circle(screen, (255, 220, 60), d, 6)

    # Confidence badges
    font = pygame.font.SysFont("monospace", 14, bold=True)
    for i, c in enumerate(confs):
        col  = (90, 220, 110) if c >= 0.90 else (240, 210, 80) if c >= 0.70 else (235, 90, 90)
        txt  = font.render(f"P{i+1} p={c:.2f}", True, col)
        screen.blit(txt, (8, 8 + i * 20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlockBlaster Android auto-player.")
    parser.add_argument("--serial",  default=None, help="ADB device serial")
    parser.add_argument("--display", action="store_true", help="Show pygame preview window")
    args = parser.parse_args()
    run(serial=args.serial, display=args.display)
