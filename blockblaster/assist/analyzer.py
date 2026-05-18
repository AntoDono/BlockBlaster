"""Background frame-analysis worker for the assist GUI.

Decouples expensive computer-vision work (``scan_board``,
``recognize_queue_with_confidence``) and the ValueNet policy
(``Advisor.suggest``) from the pygame render thread.

The worker re-reads the device's latest frame in its own loop and publishes
a thread-safe snapshot ``(frame_id, board_grid, queue, confidences,
suggestion)``.  The render thread simply reads the latest snapshot — it
never blocks on CV/CNN work — so the GUI stays smooth even when ADB
capture is slow.

Calibration mutations made on the main thread (``cfg.grid = new_box``)
are picked up automatically: the worker reads the shared
``CalibrationConfig`` instance every iteration.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from blockblaster.assist.advisor import Advisor, Suggestion
from blockblaster.assist.calibration import CalibrationConfig
from blockblaster.assist.piece_recognizer import PieceRecognizer
from blockblaster.assist.scanner import scan_board
from blockblaster.control.device import Device
from blockblaster.game.pieces import Piece


class AnalysisWorker:
    """Runs scan + recognise + advise on a daemon thread.

    The worker takes a reference to a :class:`CalibrationConfig`; the GUI
    thread can mutate ``cfg.grid`` / ``cfg.queue`` directly and the next
    worker iteration will use the new values.
    """

    _IDLE_SLEEP = 0.005   # 5 ms — keep the worker responsive without busy-looping

    def __init__(
        self,
        device: Device,
        cfg: CalibrationConfig,
        recognizer: PieceRecognizer,
        advisor: Advisor,
    ) -> None:
        self._device     = device
        self._cfg        = cfg
        self._recognizer = recognizer
        self._advisor    = advisor

        # Published snapshot — protected by ``_lock``.
        self._lock                 = threading.Lock()
        self._snap_frame_id: int   = -1
        self._snap_board_grid      = np.zeros((8, 8), dtype=bool)
        self._snap_queue: list[Optional[Piece]] = []
        self._snap_confidences: list[float]     = []
        self._snap_suggestion: Optional[Suggestion] = None

        self._running  = False
        self._paused   = False
        self._thread: Optional[threading.Thread] = None

    # ── Public API ──────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="assist-analyzer", daemon=True,
        )
        self._thread.start()

    def pause(self) -> None:
        """Freeze the queue CNN + advisor until :meth:`resume` is called.

        Board scanning keeps running — the published snapshot still gets a
        fresh ``board_grid`` every iteration so the recon panel can show
        the ghost piece drifting into position during a visual-servo
        placement.  Only ``queue`` / ``confidences`` / ``suggestion`` are
        held at their last-known values.
        """
        self._paused = True

    def resume(self) -> None:
        """Re-enable queue recognition + advisor after a :meth:`pause`."""
        self._paused = False

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def snapshot(self) -> tuple[
        int,
        np.ndarray,
        list[Optional[Piece]],
        list[float],
        Optional[Suggestion],
    ]:
        """Return the latest ``(frame_id, board_grid, queue, confs, suggestion)``.

        The returned ``board_grid`` is a reference to the worker's most
        recently published array — the worker only ever publishes fresh
        arrays (never mutates in place), so it's safe to hold a reference
        across multiple render frames.
        """
        with self._lock:
            return (
                self._snap_frame_id,
                self._snap_board_grid,
                self._snap_queue,
                self._snap_confidences,
                self._snap_suggestion,
            )

    # ── Internal ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        last_seen_id = -1
        while self._running:
            frame, frame_id = self._device.get_latest_with_id()
            if frame is None or frame_id == last_seen_id:
                time.sleep(self._IDLE_SLEEP)
                continue

            cfg = self._cfg
            grid_box  = cfg.grid
            queue_box = cfg.queue

            board_grid = self._snap_board_grid
            if grid_box is not None and grid_box.is_valid():
                board_grid = scan_board(frame, grid_box)

            if self._paused:
                # Visual-servo in flight: keep refreshing the board (so the
                # recon panel can show the ghost moving) but freeze the
                # queue / confidences / suggestion at their last values.
                with self._lock:
                    self._snap_frame_id   = frame_id
                    self._snap_board_grid = board_grid
                last_seen_id = frame_id
                continue

            queue: list[Optional[Piece]] = []
            confidences: list[float]     = []
            if queue_box is not None and queue_box.is_valid():
                results = self._recognizer.recognize_queue_with_confidence(
                    frame, queue_box,
                )
                queue       = [p for p, _ in results]
                confidences = [c for _, c in results]

            suggestion = self._advisor.suggest(board_grid, queue) if queue else None

            with self._lock:
                self._snap_frame_id    = frame_id
                self._snap_board_grid  = board_grid
                self._snap_queue       = queue
                self._snap_confidences = confidences
                self._snap_suggestion  = suggestion

            last_seen_id = frame_id
