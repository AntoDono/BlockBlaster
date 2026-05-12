"""3-piece beam-search lookahead policy with epsilon-exploration.

The value net is trained against *shaped* MC targets G_t - Phi(s_t), so it
predicts V_F(s) = V*(s) - Phi(s).  To recover the original optimal policy we
add Phi(s') back at action-selection time:

    a* = argmax_a [ V_F(s') + Phi(s') ]

The policy uses beam search across the full 3-piece queue:

  Depth 0->1  place piece A at all legal positions, score V*(board_A, [B,C]),
              keep top BEAM_WIDTH candidates.
  Depth 1->2  expand each beam with piece B, score V*(board_AB, [C]),
              keep top BEAM_WIDTH.
  Depth 2->3  expand each beam with piece C, score V*(board_ABC, []),
              return first_action of the global argmax.

Intermediate states are scored with the remaining unplaced pieces as queue
context — consistent with how the net was trained.  Dead-end beams (no legal
position for the next piece) are scored as V=0.  Distinct orderings of the
3 queue pieces are tried and the best first_action across all orderings is
returned.

See `blockblaster/game/potential.py` and `blockblaster/train/dataset.py`.
"""

from __future__ import annotations

import random
from itertools import permutations
from typing import Optional

import numpy as np
import torch

import param
from blockblaster.game.board import Board
from blockblaster.game.env import BlockBlastEnv
from blockblaster.game.pieces import Piece
from blockblaster.game.potential import board_potential
from blockblaster.model.encoder import encode_state
from blockblaster.model.value_net import ValueNet


# ---------------------------------------------------------------------------
# Grid helpers (operate on raw np.ndarray to avoid env-clone overhead)
# ---------------------------------------------------------------------------

def _legal_positions(grid: np.ndarray, piece: Piece) -> list[tuple[int, int]]:
    """Return all (row, col) top-left positions where `piece` fits on `grid`."""
    n = grid.shape[0]
    pr, pc = piece.rows, piece.cols
    if pr > n or pc > n:
        return []
    valid = np.ones((n - pr + 1, n - pc + 1), dtype=bool)
    for dr, dc in piece.cells:
        valid &= grid[dr : n - pr + 1 + dr, dc : n - pc + 1 + dc] == 0
    rs, cs = np.where(valid)
    return list(zip(rs.tolist(), cs.tolist()))


def _place_and_clear(grid: np.ndarray, piece: Piece, row: int, col: int) -> np.ndarray:
    """Return a new grid with `piece` placed at (row, col) and full lines cleared."""
    g = grid.copy()
    for dr, dc in piece.cells:
        g[row + dr, col + dc] = 1
    full_rows = np.where(g.all(axis=1))[0]
    full_cols = np.where(g.all(axis=0))[0]
    if len(full_rows) or len(full_cols):
        g[full_rows, :] = 0
        g[:, full_cols] = 0
    return g


# ---------------------------------------------------------------------------
# Beam search helpers
# ---------------------------------------------------------------------------

def _batch_score(
    grids: list[np.ndarray],
    queues: list[list[Piece]],
    net: ValueNet,
    net_device: torch.device,
    chunk_size: int,
) -> list[float]:
    """Return V*(grid, queue) = V_F(grid) + Phi(grid) for each (grid, queue) pair.

    Encodes boards in chunks to bound GPU memory.  The queue is passed to
    the encoder so intermediate states use the correct remaining-piece context.
    """
    if not grids:
        return []
    board_obj = Board()
    tensors: list[torch.Tensor] = []
    phis: list[float] = []
    for g, q in zip(grids, queues):
        board_obj.grid = g
        tensors.append(encode_state(board_obj, q))
        phis.append(board_potential(g))
    value_chunks: list[torch.Tensor] = []
    for start in range(0, len(tensors), chunk_size):
        batch = torch.stack(tensors[start : start + chunk_size], dim=0).to(net_device)
        value_chunks.append(net.predict(batch))
    values = torch.cat(value_chunks)
    phi_t = torch.tensor(phis, dtype=values.dtype, device=values.device)
    return (values + phi_t).tolist()


def _top_k(
    candidates: list,
    scores: list[float],
    k: int,
) -> list:
    """Return the top-k candidates by descending score."""
    if len(candidates) <= k:
        return list(candidates)
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in order[:k]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# BeamEntry type alias (for clarity):
#   (first_action: tuple[int,int,int],
#    current_grid: np.ndarray,
#    remaining_pieces: list[Piece])
_BeamEntry = tuple


def select_action(
    env: BlockBlastEnv,
    net: Optional[ValueNet],
    epsilon: float = 0.0,
    device: str | None = None,
) -> tuple[int, int, int]:
    """Choose a (slot, row, col) action via beam-search 3-piece lookahead.

    Strategy:
      - With probability `epsilon` or if `net` is None: uniform random from
        the 1-step legal action set.
      - Otherwise: beam search over all distinct orderings of the queue.
        At each of the 3 depths, candidates are scored as V*(board, remaining)
        and only the top BEAM_WIDTH are expanded at the next depth.
        Dead-end beams (no legal position for the next piece) score V=0.
    """
    actions = env.legal_actions()
    if not actions:
        raise RuntimeError("select_action called on a terminal state (no legal actions)")

    if net is None or random.random() < epsilon:
        return random.choice(actions)

    net_device = next(net.parameters()).device
    queue = env.queue
    grid = env.board.grid
    n_pieces = len(queue)
    chunk = param.LOOKAHEAD_MAX_BATCH
    width = param.BEAM_WIDTH

    # Build distinct orderings, deduped by piece-id tuple.
    seen_orderings: set[tuple[int, ...]] = set()
    distinct_orderings: list[tuple[int, ...]] = []
    for perm in permutations(range(n_pieces)):
        pid_key = tuple(queue[i].piece_id for i in perm)
        if pid_key not in seen_orderings:
            seen_orderings.add(pid_key)
            distinct_orderings.append(perm)

    # ------------------------------------------------------------------
    # Depth 0 -> 1: expand the first piece of every distinct ordering.
    # ------------------------------------------------------------------
    # Each candidate: (first_action, grid_after_depth1, remaining_pieces)
    d1_cands: list[_BeamEntry] = []
    for ordering in distinct_orderings:
        p0 = queue[ordering[0]]
        remaining = [queue[ordering[i]] for i in range(1, n_pieces)]
        for r, c in _legal_positions(grid, p0):
            d1_cands.append(
                ((ordering[0], r, c), _place_and_clear(grid, p0, r, c), remaining)
            )

    if not d1_cands:
        # No ordering has a legal first move — fall back to random.
        return random.choice(actions)

    d1_scores = _batch_score(
        [c[1] for c in d1_cands],
        [c[2] for c in d1_cands],
        net, net_device, chunk,
    )
    beams: list[_BeamEntry] = _top_k(d1_cands, d1_scores, width)

    # Track the running best for dead-end beams (scored as V=0).
    best_score: float = -float("inf")
    best_action: tuple[int, int, int] = actions[0]

    # ------------------------------------------------------------------
    # Depth 1 -> 2: expand the second piece on surviving beams.
    # ------------------------------------------------------------------
    d2_cands: list[_BeamEntry] = []
    for first_action, g1, remaining in beams:
        if not remaining:
            # Queue had only 1 piece; treat this beam as a leaf.
            score = _batch_score([g1], [[]], net, net_device, chunk)[0]
            if score > best_score:
                best_score = score
                best_action = first_action
            continue
        p1 = remaining[0]
        positions = _legal_positions(g1, p1)
        if not positions:
            # Dead-end: score = 0.
            if 0.0 > best_score:
                best_score = 0.0
                best_action = first_action
            continue
        rem2 = remaining[1:]
        for r, c in positions:
            d2_cands.append((first_action, _place_and_clear(g1, p1, r, c), rem2))

    if not d2_cands:
        return best_action  # all depth-1 beams dead-ended or were leaves

    d2_scores = _batch_score(
        [c[1] for c in d2_cands],
        [c[2] for c in d2_cands],
        net, net_device, chunk,
    )
    beams = _top_k(d2_cands, d2_scores, width)

    # ------------------------------------------------------------------
    # Depth 2 -> 3 (final): expand the third piece, pick global argmax.
    # ------------------------------------------------------------------
    d3_cands: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for first_action, g2, remaining in beams:
        if not remaining:
            # Queue had only 2 pieces; treat this beam as a leaf.
            score = _batch_score([g2], [[]], net, net_device, chunk)[0]
            if score > best_score:
                best_score = score
                best_action = first_action
            continue
        p2 = remaining[0]
        positions = _legal_positions(g2, p2)
        if not positions:
            if 0.0 > best_score:
                best_score = 0.0
                best_action = first_action
            continue
        for r, c in positions:
            d3_cands.append((first_action, _place_and_clear(g2, p2, r, c)))

    if d3_cands:
        d3_scores = _batch_score(
            [c[1] for c in d3_cands],
            [[] for _ in d3_cands],
            net, net_device, chunk,
        )
        for i, (first_action, _) in enumerate(d3_cands):
            if d3_scores[i] > best_score:
                best_score = d3_scores[i]
                best_action = first_action

    return best_action
