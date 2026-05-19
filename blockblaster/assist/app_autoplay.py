"""Auto-play move execution for the assist GUI.

Decides — once per fresh analysis frame — whether to drive a closed-loop
visual-servo placement on the connected device, and persists timing /
"already-executed-this-frame" state on :class:`AppState`.
"""

from __future__ import annotations

import threading
from typing import Optional

import pygame

from blockblaster.assist.advisor import Suggestion
from blockblaster.assist.analyzer import AnalysisWorker
from blockblaster.assist.app_state import AppState
from blockblaster.config.params import (
    AUTO_CONF_THRESHOLD,
    AUTO_POST_PLACE_MS,
    AUTO_SERVO_BUDGET_MS,
)
from blockblaster.control.coords import piece_anchor_px, slot_center_px
from blockblaster.control.device import Device
from blockblaster.control.servo import GRAB_Y_NUDGE_PX, place


def _run_servo_safely(
    device: Device,
    state:  AppState,
    analyzer: AnalysisWorker,
    suggestion: Suggestion,
    frame_w: int,
    frame_h: int,
) -> None:
    """Run :func:`servo.place` on a worker thread; disable auto-play on hard failure.

    Always unfreezes the GUI and resumes the analyzer in a ``finally`` block
    so a crash mid-servo can never leave the assist window pinned to a stale
    snapshot.
    """
    try:
        ok = place(
            device=device,
            cfg=state.cfg,
            suggestion=suggestion,
            frame_w=frame_w,
            frame_h=frame_h,
            state=state,
        )
        print(f"[auto] servo: {'ok' if ok else 'FAIL'}")
    except Exception as exc:
        print(f"[auto] servo crashed, disabling auto-play: {exc}")
        state.auto_enabled = False
    finally:
        state.servo_active       = False
        state.servo_detection    = None
        state.frozen_suggestion  = None
        state.frozen_queue       = []
        state.frozen_confidences = []
        analyzer.resume()


def maybe_execute_auto_swipe(
    *,
    device: Device,
    state: AppState,
    analyzer: AnalysisWorker,
    suggestion: Optional[Suggestion],
    queue: list,
    queue_confidences: list[float],
    analysis_changed: bool,
    analysis_frame_id: int,
) -> None:
    """Kick off an auto-play servo placement if every gate is satisfied.

    Mutates ``state.auto_last_executed_frame``, ``state.auto_busy_until``,
    and ``state.auto_enabled`` (disabled on servo crash).
    """
    # Each gate logs *once* per consecutive run of failures via
    # state.auto_last_gate_reason, so the terminal isn't drowned in
    # repeats but every state change is visible.  Helps diagnose
    # "the servo stopped firing — why?" without adding prints to every
    # caller.
    def _gate(reason: str) -> None:
        last = getattr(state, "auto_last_gate_reason", None)
        if last != reason:
            print(f"[auto] gated: {reason}")
            state.auto_last_gate_reason = reason

    if not state.auto_enabled:
        _gate("auto disabled")
        return
    if suggestion is None:
        _gate("no suggestion from advisor")
        return
    if state.cfg.grid is None or not state.cfg.grid.is_valid():
        _gate("grid not calibrated")
        return
    if state.cfg.queue is None or not state.cfg.queue.is_valid():
        _gate("queue not calibrated")
        return
    if not analysis_changed:
        return  # quiet: fires every tick between analyzer updates
    if analysis_frame_id == state.auto_last_executed_frame:
        return  # quiet: same frame as the dispatch we just made
    if not queue_confidences:
        _gate("no queue confidences yet")
        return
    if not all(c >= AUTO_CONF_THRESHOLD for c in queue_confidences):
        low = [f"{c:.2f}" for c in queue_confidences]
        _gate(f"queue CNN confidence below {AUTO_CONF_THRESHOLD}: {low}")
        return
    if pygame.time.get_ticks() < state.auto_busy_until:
        return  # quiet: cooldown after previous servo
    # All gates passed — clear the sticky reason so the next stall logs.
    state.auto_last_gate_reason = None

    try:
        slot_cx, slot_cy = slot_center_px(state.cfg.queue, suggestion.slot)
        # Mirror the servo grab-point nudge so the on-screen arrow
        # shows where the finger actually touches down.
        src = (slot_cx, slot_cy - GRAB_Y_NUDGE_PX)
        dst = piece_anchor_px(
            state.cfg.grid, suggestion.piece, suggestion.row, suggestion.col,
        )
        try:
            dev_w, dev_h = device.screen_size()
            scale_ok = (state.frame_w, state.frame_h) == (dev_w, dev_h)
            print(
                f"[auto] {suggestion.piece.name} slot={suggestion.slot+1} "
                f"→ row={suggestion.row+1} col={suggestion.col+1}  "
                f"servo {src} → {dst}  "
                f"frame={state.frame_w}x{state.frame_h}  device={dev_w}x{dev_h}  "
                f"{'match' if scale_ok else 'rescale'}"
            )
        except Exception:
            print(
                f"[auto] {suggestion.piece.name} slot={suggestion.slot+1} "
                f"→ row={suggestion.row+1} col={suggestion.col+1}  "
                f"servo {src} → {dst}"
            )

        t_now = pygame.time.get_ticks()
        # Keep the on-screen arrow in frame-pixel space.
        state.last_swipe = (src, dst, t_now, AUTO_SERVO_BUDGET_MS)

        # Freeze the queue CNN + advisor output to what we saw at dispatch
        # time, and pause the analyzer's queue/advisor pass so mid-swipe
        # CNN reads don't reshape the on-screen suggestion.  Board scanning
        # keeps running — the recon panel needs it to show the held piece's
        # solid render drifting into place.
        state.frozen_suggestion  = suggestion
        state.frozen_queue       = list(queue)
        state.frozen_confidences = list(queue_confidences)
        state.servo_active = True
        analyzer.pause()

        threading.Thread(
            target=_run_servo_safely,
            args=(device, state, analyzer, suggestion,
                  state.frame_w, state.frame_h),
            name="auto-servo",
            daemon=True,
        ).start()
        state.auto_last_executed_frame = analysis_frame_id
        state.auto_busy_until = t_now + AUTO_SERVO_BUDGET_MS + AUTO_POST_PLACE_MS
    except Exception as exc:
        print(f"[auto] servo dispatch failed, disabling auto-play: {exc}")
        state.auto_enabled = False
        state.servo_active = False
        state.frozen_suggestion  = None
        state.frozen_queue       = []
        state.frozen_confidences = []
        analyzer.resume()
