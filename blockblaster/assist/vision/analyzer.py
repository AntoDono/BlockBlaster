"""Background worker: detect interactables, scan board, recognise pieces, advise."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from blockblaster.assist.advisor import Advisor, Suggestion
from blockblaster.assist.vision.detection import (
    Element,
    detect_interactables,
    estimate_background_bgr,
    split_roles,
)
from blockblaster.assist.vision.piece_recognizer import PieceRecognizer, pad_to_slot
from blockblaster.assist.vision.scanner import BOARD_SIZE, scan_board
from blockblaster.control.device import Device
from blockblaster.game.pieces import Piece
from blockblaster.piece_cnn.config import INPUT_SIZE


@dataclass
class PieceDetection:
    piece: Optional[Piece]
    bbox: tuple[int, int, int, int]
    confidence: float
    cnn_input: Optional[np.ndarray] = None


@dataclass
class ReconSnapshot:
    frame_id: int = -1
    elements: list[Element] = field(default_factory=list)
    board_grid: np.ndarray = field(
        default_factory=lambda: np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    )
    board_bbox: Optional[tuple[int, int, int, int]] = None
    pieces: list[PieceDetection] = field(default_factory=list)
    suggestion: Optional[Suggestion] = None


class AnalysisWorker:
    _IDLE_SLEEP = 0.005
    # A changed board must persist this many consecutive frames before it counts
    # as a real change (debounces scan flicker during a drop / clear animation).
    _STABLE_FRAMES = 2

    def __init__(
        self,
        device: Device,
        recognizer: PieceRecognizer,
        advisor: Advisor,
        diff_tracker: Optional[object] = None,
    ) -> None:
        self._device     = device
        self._recognizer = recognizer
        self._advisor    = advisor
        self._diff_tracker = diff_tracker

        self._lock     = threading.Lock()
        self._snap     = ReconSnapshot()
        self._running  = False
        self._thread: Optional[threading.Thread] = None

        # Suggestion latch: a placement is held until the *board* changes, so a
        # piece being lifted/dragged (which only changes the tray) never causes a
        # premature re-suggest. The board change is debounced for stability.
        self._held_suggestion: Optional[Suggestion] = None
        self._held_board: Optional[np.ndarray] = None
        self._cand_board: Optional[np.ndarray] = None
        self._cand_count: int = 0
        self._confirmed_board: Optional[np.ndarray] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="assist-analyzer", daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def snapshot(self) -> ReconSnapshot:
        with self._lock:
            return self._snap

    def _loop(self) -> None:
        last_seen_id = -1
        in_event = False
        while self._running:
            frame, frame_id = self._device.get_latest_with_id()
            if frame is None:
                time.sleep(self._IDLE_SLEEP)
                continue

            # Pause analysis while a big-motion event (drop / clear animation) is
            # in progress; the board is mid-transition and would scan garbage.
            if self._diff_tracker is not None and self._diff_tracker.event_active(
                time.monotonic()
            ):
                in_event = True
                time.sleep(self._IDLE_SLEEP)
                continue

            # The instant the event clears, force a re-analysis of the current
            # frame even if its id hasn't advanced.
            if in_event:
                in_event = False
                last_seen_id = -1

            if frame_id == last_seen_id:
                time.sleep(self._IDLE_SLEEP)
                continue

            snap = self._analyze(frame, frame_id)
            with self._lock:
                self._snap = snap
            last_seen_id = frame_id

    def _analyze(self, frame: np.ndarray, frame_id: int) -> ReconSnapshot:
        elements      = detect_interactables(frame, detect_board=False)
        board, pieces = split_roles(elements)

        board_grid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
        board_bbox = None
        if board is not None:
            board_bbox = board.bbox
            board_grid = scan_board(frame, board_bbox)

        piece_dets: list[PieceDetection] = []
        if pieces:
            bg      = estimate_background_bgr(frame)
            crops   = [pad_to_slot(frame, p.bbox, bg) for p in pieces]
            results = self._recognizer.recognize_crops(crops)
            for p_elem, crop, (piece, conf) in zip(pieces, crops, results):
                cnn_input = cv2.resize(
                    crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA,
                )
                piece_dets.append(PieceDetection(piece, p_elem.bbox, conf, cnn_input))

        queue      = [pd.piece for pd in piece_dets]
        suggestion = self._resolve_suggestion(board_grid, queue)

        return ReconSnapshot(
            frame_id=frame_id,
            elements=elements,
            board_grid=board_grid,
            board_bbox=board_bbox,
            pieces=piece_dets,
            suggestion=suggestion,
        )

    def _resolve_suggestion(
        self,
        board_grid: np.ndarray,
        queue: list[Optional[Piece]],
    ) -> Optional[Suggestion]:
        """Hold the current suggestion until the (debounced) board changes.

        Lifting/dragging a piece only mutates the tray, not the board, so the
        held placement survives the whole move and is only recomputed once the
        piece actually lands (or a line clears) and the new board stabilises.
        """
        confirmed = self._update_confirmed_board(board_grid)
        if confirmed is None:
            return self._held_suggestion

        board_changed = (
            self._held_board is None
            or not np.array_equal(confirmed, self._held_board)
        )
        if not board_changed:
            return self._held_suggestion

        if any(p is not None for p in queue):
            new = self._advisor.suggest(confirmed, queue)
            if new is not None:
                self._held_suggestion = new
                self._held_board = confirmed.copy()

        return self._held_suggestion

    def _update_confirmed_board(
        self, board_grid: np.ndarray
    ) -> Optional[np.ndarray]:
        """Return the latest board grid that has been stable for N frames."""
        if self._cand_board is None or not np.array_equal(board_grid, self._cand_board):
            self._cand_board = board_grid
            self._cand_count = 1
        else:
            self._cand_count += 1

        if self._cand_count >= self._STABLE_FRAMES:
            self._confirmed_board = board_grid
        return self._confirmed_board
