"""Suggest the best next placement using a trained ValueNet.

Wraps :func:`blockblaster.agent.policy.select_action` for the assist GUI.
The advisor:

  * loads weights from ``model.pt`` (or any path) at construction,
  * accepts the *scanned* board grid + recognised queue (which may contain
    ``None`` slots if recognition failed),
  * returns the policy's chosen ``(slot, row, col)`` action plus a handle
    on the chosen :class:`Piece`,
  * caches the last suggestion by (board, queue piece-ids) so we don't
    rerun beam search every frame.

If the model fails to load, or no legal move exists, ``suggest()`` returns
``None`` and ``last_error`` carries a human-readable reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import param
from blockblaster.agent.policy import select_action
from blockblaster.game.board import Board
from blockblaster.game.env import BlockBlastEnv
from blockblaster.game.pieces import Piece
from blockblaster.model.value_net import ValueNet


@dataclass(frozen=True)
class Suggestion:
    slot: int
    row: int
    col: int
    piece: Piece


class Advisor:
    """Loads a ValueNet checkpoint and produces placement suggestions."""

    def __init__(self, model_path: str | Path = "model.pt") -> None:
        self.model_path = Path(model_path)
        self.net: Optional[ValueNet] = None
        self.last_error: Optional[str] = None
        self._cache_key: Optional[tuple] = None
        self._cache_value: Optional[Suggestion] = None
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.model_path.exists():
            self.last_error = f"model not found: {self.model_path}"
            return
        try:
            net = ValueNet()
            data = torch.load(
                self.model_path,
                map_location=param.DEVICE,
                weights_only=True,
            )
            if isinstance(data, dict) and "state_dict" in data:
                net.load_state_dict(data["state_dict"])
            else:
                net.load_state_dict(data)  # type: ignore[arg-type]
            net.to(param.DEVICE)
            net.eval()
            self.net = net
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 — surface any load failure
            self.net = None
            self.last_error = f"model load failed: {exc!r}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(
        self,
        board_grid: np.ndarray,
        queue: list[Optional[Piece]],
    ) -> Optional[Suggestion]:
        """Return the policy's suggested placement, or ``None`` if unavailable.

        Slots in ``queue`` that are ``None`` (already played, or unrecognised)
        are filtered out before running the policy.  As long as at least one
        piece is recognised we still produce a suggestion; the returned
        ``Suggestion.slot`` indexes the *original* queue (not the filtered
        list) so the GUI can highlight the right slot.
        """
        if self.net is None:
            return None
        if not queue:
            return None

        # Filter out None slots and remember each surviving piece's position
        # in the original queue.
        original_slots: list[int] = [i for i, p in enumerate(queue) if p is not None]
        pieces: list[Piece] = [queue[i] for i in original_slots]
        if not pieces:
            return None

        cache_key = (
            board_grid.tobytes(),
            board_grid.shape,
            tuple((i, p.piece_id) for i, p in zip(original_slots, pieces)),
        )
        if cache_key == self._cache_key:
            return self._cache_value

        env = BlockBlastEnv.__new__(BlockBlastEnv)
        env._rng = None  # type: ignore[attr-defined]  # select_action doesn't touch the rng
        env.board = Board()
        env.board.grid = board_grid.astype(np.int8).copy()
        env.queue = list(pieces)
        env.total_score = 0.0
        env.steps = 0

        if not env.legal_actions():
            self._cache_key = cache_key
            self._cache_value = None
            return None

        try:
            slot_filtered, row, col = select_action(env, self.net, epsilon=0.0)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"select_action failed: {exc!r}"
            self._cache_key = cache_key
            self._cache_value = None
            return None

        # Map the filtered slot index back to the original queue
        original_slot = original_slots[slot_filtered]
        suggestion = Suggestion(
            slot=original_slot,
            row=row,
            col=col,
            piece=pieces[slot_filtered],
        )
        self._cache_key = cache_key
        self._cache_value = suggestion
        return suggestion
