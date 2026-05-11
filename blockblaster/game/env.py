"""Block Blast environment: orchestrates board, queue, and scoring."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import param
from blockblaster.game.board import Board
from blockblaster.game.pieces import Piece, sample_queue
from blockblaster.game.scoring import step_reward


@dataclass
class StepResult:
    reward: float
    lines_cleared: int
    done: bool


class BlockBlastEnv:
    """
    Stateful Block Blast environment.

    A "step" = placing one piece from the queue.
    When the queue is exhausted a new batch of QUEUE_SIZE pieces is drawn.
    The game ends when no remaining queued piece can be legally placed.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.board = Board()
        self.queue: list[Piece] = []
        self.total_score: float = 0.0
        self.steps: int = 0
        self._refill_queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        self.board = Board()
        self.queue = []
        self.total_score = 0.0
        self.steps = 0
        self._refill_queue()

    def step(self, slot: int, row: int, col: int) -> StepResult:
        """
        Place the piece at `queue[slot]` at (row, col) on the board.

        Returns StepResult with reward, lines cleared, and done flag.
        Raises ValueError for illegal actions.
        """
        if slot < 0 or slot >= len(self.queue):
            raise ValueError(f"Invalid queue slot {slot} (queue size {len(self.queue)})")
        piece = self.queue[slot]
        lines = self.board.place(piece, row, col)
        reward = step_reward(len(piece.cells), lines)
        self.total_score += reward
        self.steps += 1

        self.queue.pop(slot)
        if not self.queue:
            self._refill_queue()

        done = self.board.is_game_over(self.queue)
        return StepResult(reward=reward, lines_cleared=lines, done=done)

    def legal_actions(self) -> list[tuple[int, int, int]]:
        """Return list of (slot, row, col) tuples that are currently legal."""
        actions: list[tuple[int, int, int]] = []
        seen_ids: set[int] = set()
        for slot, piece in enumerate(self.queue):
            # Skip duplicate pieces in the queue to avoid redundant actions
            if piece.piece_id in seen_ids:
                continue
            seen_ids.add(piece.piece_id)
            for r, c in self.board.legal_placements(piece):
                actions.append((slot, r, c))
        return actions

    def is_over(self) -> bool:
        return self.board.is_game_over(self.queue)

    def clone(self) -> "BlockBlastEnv":
        """Return a deep copy of the environment (cheap board copy)."""
        env = BlockBlastEnv.__new__(BlockBlastEnv)
        env._rng = random.Random()
        env._rng.setstate(self._rng.getstate())
        env.board = self.board.clone()
        env.queue = list(self.queue)
        env.total_score = self.total_score
        env.steps = self.steps
        return env

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refill_queue(self) -> None:
        self.queue = sample_queue(param.QUEUE_SIZE, self._rng)
