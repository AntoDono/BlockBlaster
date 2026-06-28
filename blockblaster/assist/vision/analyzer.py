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

    def __init__(
        self,
        device: Device,
        recognizer: PieceRecognizer,
        advisor: Advisor,
    ) -> None:
        self._device     = device
        self._recognizer = recognizer
        self._advisor    = advisor

        self._lock     = threading.Lock()
        self._snap     = ReconSnapshot()
        self._running  = False
        self._thread: Optional[threading.Thread] = None

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
        while self._running:
            frame, frame_id = self._device.get_latest_with_id()
            if frame is None or frame_id == last_seen_id:
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
        suggestion = (
            self._advisor.suggest(board_grid, queue)
            if any(p is not None for p in queue) else None
        )

        return ReconSnapshot(
            frame_id=frame_id,
            elements=elements,
            board_grid=board_grid,
            board_bbox=board_bbox,
            pieces=piece_dets,
            suggestion=suggestion,
        )
