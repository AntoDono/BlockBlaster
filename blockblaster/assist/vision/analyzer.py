"""Background worker: detect interactables, scan board, recognise pieces, advise."""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from blockblaster.assist.advisor import Advisor, Suggestion
from blockblaster.assist.vision.detection import (
    Element,
    detect_interactables,
    estimate_background_bgr,
    reset_board_cache,
    split_roles,
)
from blockblaster.assist.vision.piece_recognizer import PieceRecognizer, pad_to_slot
from blockblaster.assist.vision.scanner import BOARD_SIZE, scan_board
from blockblaster.control.device import Device
from blockblaster.game.pieces import Piece
from blockblaster.piece_cnn.config import INPUT_SIZE

if TYPE_CHECKING:
    from blockblaster.assist.vision.frame_diff import FrameDiffTracker


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
    _STABLE_FRAMES = 10

    def __init__(
        self,
        device: Device,
        recognizer: PieceRecognizer,
        advisor: Advisor,
        diff_tracker: Optional[FrameDiffTracker] = None,
    ) -> None:
        self._device     = device
        self._recognizer = recognizer
        self._advisor    = advisor
        self._diff_tracker = diff_tracker

        self._lock     = threading.Lock()
        self._snap     = ReconSnapshot()
        self._running  = False
        self._thread: Optional[threading.Thread] = None

        # Suggestion latch keyed on the combined (board, queue) state, debounced
        # for stability. A dragged piece keeps the board changing every frame so
        # the state never confirms (suggestion held, no flicker); once a
        # placement settles OR a new piece set is dealt, the state confirms and
        # the suggestion is recomputed — so it never points at a stale piece.
        self._held_suggestion: Optional[Suggestion] = None
        self._held_board: Optional[np.ndarray] = None
        self._held_queue: Optional[tuple] = None
        self._cand_board: Optional[np.ndarray] = None
        self._cand_queue: Optional[tuple] = None
        self._cand_count: int = 0
        self._confirmed_board: Optional[np.ndarray] = None
        self._confirmed_queue: Optional[tuple] = None
        self._pause_until: float = 0.0
        self._force_reanalyze: bool = False

        # Manual board-region override (frame px (x,y,w,h)); when set, it is used
        # for board scanning/suggestion instead of the auto-detected board.
        self._board_override: Optional[tuple[int, int, int, int]] = None

    def set_board_override(
        self, bbox: Optional[tuple[int, int, int, int]]
    ) -> None:
        """Force the board region (or pass ``None`` to resume auto-detection).

        Resets the debounce so a stale confirmed state can't linger past the
        edit.
        """
        with self._lock:
            if bbox == self._board_override:
                return
            self._board_override = bbox
            self._reset_latches()

    def pause_until(self, deadline: float) -> None:
        """Extend pause to at least ``deadline`` (``time.monotonic()``)."""
        with self._lock:
            self._pause_until = max(self._pause_until, deadline)

    def set_pause_until(self, deadline: float) -> None:
        """Set pause deadline exactly (e.g. shorten after servo ok)."""
        with self._lock:
            self._pause_until = deadline

    def analysis_paused(self, now: Optional[float] = None) -> bool:
        return (now if now is not None else time.monotonic()) < self._pause_until

    def discard_suggestion(self) -> None:
        """Drop the held/snapshot suggestion without stopping analysis."""
        with self._lock:
            self._held_suggestion = None
            self._held_board = None
            self._held_queue = None
            self._snap = dataclasses.replace(self._snap, suggestion=None)

    def reset_debounce(self) -> None:
        """Clear debounce state so the next analysis pass starts fresh."""
        with self._lock:
            self._cand_board = None
            self._cand_queue = None
            self._cand_count = 0
            self._confirmed_board = None
            self._confirmed_queue = None

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
        """Return a shallow copy of the latest snapshot, safe to read from any thread."""
        with self._lock:
            return dataclasses.replace(self._snap)

    def reset_analysis(self) -> None:
        """Drop all latched state so the next frame is analysed fresh.

        Clears the held suggestion, candidate/confirmed board, advisor cache,
        detected-board cache, and the published snapshot so the UI reflects the
        reset immediately.
        """
        with self._lock:
            self._reset_latches()
            self._snap = dataclasses.replace(
                self._snap,
                suggestion=None,
                board_bbox=None,
                board_grid=np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool),
                pieces=[],
            )
            self._force_reanalyze = True
        self._advisor.clear_cache()
        reset_board_cache()

    def _reset_latches(self) -> None:
        self._held_suggestion = None
        self._held_board = None
        self._held_queue = None
        self._cand_board = None
        self._cand_queue = None
        self._cand_count = 0
        self._confirmed_board = None
        self._confirmed_queue = None
        self._pause_until = 0.0

    def _loop(self) -> None:
        last_seen_id = -1
        in_event = False
        while self._running:
            frame, frame_id = self._device.get_latest_with_id()
            if frame is None:
                time.sleep(self._IDLE_SLEEP)
                continue

            if time.monotonic() < self._pause_until:
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

            with self._lock:
                if self._force_reanalyze:
                    self._force_reanalyze = False
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
        if self._board_override is not None:
            board_bbox = self._board_override
            board_grid = scan_board(frame, board_bbox)
        elif board is not None:
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
        """Hold the suggestion until the (debounced) board *or* queue changes.

        A dragged piece keeps the board scan changing every frame, so the state
        never stabilises and the held placement survives the whole move. Once a
        placement settles or a new piece set is dealt, the confirmed state
        differs from what the suggestion was computed against → recompute. This
        also stops the suggestion from referencing a stale/absent piece.
        """
        queue_ids = tuple(p.piece_id if p is not None else None for p in queue)
        confirmed = self._update_confirmed_state(board_grid, queue_ids)
        if confirmed is None:
            return self._held_suggestion
        cboard, cqueue = confirmed

        changed = (
            self._held_board is None
            or self._held_queue != cqueue
            or not np.array_equal(cboard, self._held_board)
        )
        if not changed:
            return self._held_suggestion

        if any(p is not None for p in queue):
            new = self._advisor.suggest(cboard, queue)
            if new is not None:
                self._held_suggestion = new
                self._held_board = cboard.copy()
                self._held_queue = cqueue

        return self._held_suggestion

    def _update_confirmed_state(
        self, board_grid: np.ndarray, queue_ids: tuple,
    ) -> Optional[tuple[np.ndarray, tuple]]:
        """Return the (board, queue) state once stable for N consecutive frames."""
        if (
            self._cand_board is None
            or self._cand_queue != queue_ids
            or not np.array_equal(board_grid, self._cand_board)
        ):
            self._cand_board = board_grid
            self._cand_queue = queue_ids
            self._cand_count = 1
        else:
            self._cand_count += 1

        if self._cand_count >= self._STABLE_FRAMES:
            self._confirmed_board = board_grid
            self._confirmed_queue = queue_ids
        if self._confirmed_board is None:
            return None
        return self._confirmed_board, self._confirmed_queue
