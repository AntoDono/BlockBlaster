"""3-piece feasibility-first placement suggester backed by a trained ValueNet.

The advisor's job is the same as the training policy: pick the best
*first move* across all tray pieces.  It uses the shared lookahead in
:mod:`blockblaster.game.lookahead`, which:

* Simulates each candidate placement with row/column clears, then expands
  the remaining tray pieces.
* Treats a *feasible* first move as one that admits a full sequence
  placing every tray piece — these are strictly preferred.
* Tie-breaks feasible moves by ``r_0 + γ·r_1 + γ²·r_2 + γ³·V*(s_3)``.
* Falls back to the highest-reward terminal path when no feasible move
  exists (best-of-a-bad-situation).

On cramped boards (any tray piece has few legal positions) **every
distinct first move is kept** at the top of the search so its feasibility
is always evaluated — important on late-game boards where one of three
pieces decides survival.  On sparse boards every first move is trivially
feasible, so the same beam-bounded search the training policy uses is
sufficient.  In both regimes deeper expansions beam-prune to keep the
call cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import param
from blockblaster.game.board import legal_positions_grid
from blockblaster.game.lookahead import PlannedStep, search_first_move
from blockblaster.game.pieces import Piece
from blockblaster.model.value_net import ValueNet


# When every tray piece has at least this many legal positions on the
# current board, feasibility of fitting all three is overwhelmingly likely;
# the cheap beam search is sufficient and the value-net scoring stays fast.
# Below the threshold the board is cramped enough that the *only* feasible
# first move may not survive a top-K cut, so we keep every first move.
_DENSE_BOARD_POS_THRESHOLD = 10


@dataclass(frozen=True)
class Suggestion:
    """A single placement plus the full lookahead plan that motivated it.

    ``slot``/``row``/``col``/``piece`` describe the move the advisor wants
    the controller to execute next.  ``plan`` is the ordered sequence of
    placements the search expects to follow — ``plan[0]`` always matches
    the chosen move; ``plan[1:]`` are the planned follow-ups.  The plan
    stops at the last successfully simulated placement when ``terminal``
    is set (the 3-piece window contains an unavoidable dead-end).
    """

    slot: int
    row: int
    col: int
    piece: Piece
    plan: tuple[PlannedStep, ...] = ()
    score: float = 0.0
    terminal: bool = False


class Advisor:
    """Loads a ValueNet checkpoint and produces a feasibility-first suggestion."""

    def __init__(self, model_path: str | Path = "model.pt") -> None:
        self.model_path = Path(model_path)
        self.net: Optional[ValueNet] = None
        self.last_error: Optional[str] = None
        self._cache_key: Optional[tuple] = None
        self._cache_value: Optional[Suggestion] = None
        self._load()

    def clear_cache(self) -> None:
        """Forget the last (board, queue) → suggestion result."""
        self._cache_key = None
        self._cache_value = None

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
        except Exception as exc:  # noqa: BLE001
            self.net = None
            self.last_error = f"model load failed: {exc!r}"

    def suggest(
        self,
        board_grid: np.ndarray,
        queue: list[Optional[Piece]],
    ) -> Optional[Suggestion]:
        """Return the best first move under 3-piece feasibility-first lookahead.

        ``queue`` is the assist tray with ``None`` for any slot where piece
        detection dropped; only present slots are searched.  The returned
        ``Suggestion.slot`` is the original tray index (0/1/2), preserved
        even when intermediate slots are missing.
        """
        if self.net is None or not queue:
            return None

        tray: list[tuple[int, Piece]] = [
            (i, p) for i, p in enumerate(queue) if p is not None
        ]
        if not tray:
            return None

        grid = board_grid.astype(np.int8, copy=False)
        cache_key = (
            grid.tobytes(),
            grid.shape,
            tuple((slot, p.piece_id) for slot, p in tray),
        )
        if cache_key == self._cache_key:
            return self._cache_value

        # Cheap heuristic: only burn the per-first-move feasibility budget
        # on cramped boards.  On a sparse board, every piece has many legal
        # positions and a top-K beam never prunes the only feasible move.
        keep_all = any(
            len(legal_positions_grid(grid, p)) < _DENSE_BOARD_POS_THRESHOLD
            for _, p in tray
        )

        try:
            result = search_first_move(
                grid, tray, self.net,
                beam_width=param.BEAM_WIDTH,
                keep_all_first_moves=keep_all,
                chunk_size=param.LOOKAHEAD_MAX_BATCH,
            )
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"scoring failed: {exc!r}"
            self._cache_key = cache_key
            self._cache_value = None
            return None

        if result is None:
            self._cache_key = cache_key
            self._cache_value = None
            return None

        suggestion = Suggestion(
            slot=result.slot,
            row=result.row,
            col=result.col,
            piece=result.piece,
            plan=result.plan,
            score=result.score,
            terminal=result.terminal,
        )
        self._cache_key = cache_key
        self._cache_value = suggestion
        return suggestion
