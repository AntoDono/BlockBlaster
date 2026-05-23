"""3-piece beam-search lookahead policy with epsilon-exploration.

The value net is trained against *shaped* MC targets G_t - Phi(s_t), so it
predicts V_F(s) = V*(s) - Phi(s).  To recover the original optimal policy we
add Phi(s') back at action-selection time:

    V*(s) = V_F(s) + Phi(s)

The beam search scores each candidate sequence as the **true discounted
return** it would produce:

    score = r_0 + γ·r_1 + γ²·r_2 + γ³·V*(s_3)

where r_k is the immediate reward earned by placing the k-th piece (cells
placed + line-clear bonuses, per `blockblaster.game.scoring`).  Earlier
versions of this policy dropped the per-placement rewards and scored only
V*(s_3), which under-weighted line-clears inside the 3-piece window — a
disaster on cramped late-game boards where clearing *now* is literally
survival.  See `docs/policy.md` for the full rationale.

The policy uses beam search across the full 3-piece queue:

  Depth 0->1  place piece A at all legal positions, score
              r_0 + γ·V*(board_A, [B,C]), keep top BEAM_WIDTH.
  Depth 1->2  expand each beam with piece B, score
              r_0 + γ·r_1 + γ²·V*(board_AB, [C]), keep top BEAM_WIDTH.
  Depth 2->3  expand each beam with piece C, score
              r_0 + γ·r_1 + γ²·r_2 + γ³·V*(board_ABC, []),
              return first_action of the global argmax.

Intermediate states are scored with the remaining unplaced pieces as queue
context — consistent with how the net was trained.  Dead-end beams (no legal
position for the next piece) take their accumulated reward so far with V=0
(terminal state).  Distinct orderings of the 3 queue pieces are tried and
the best first_action across all orderings is returned.

See `blockblaster/game/potential.py`, `blockblaster/game/scoring.py`,
and `blockblaster/train/dataset.py`.
"""

from __future__ import annotations

import math
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
from blockblaster.game.scoring import step_reward
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


def _place_and_clear(
    grid: np.ndarray, piece: Piece, row: int, col: int
) -> tuple[np.ndarray, int]:
    """Place `piece` at (row, col), clear full lines.  Returns (new_grid, lines_cleared).

    `lines_cleared` is the total number of rows + cols cleared by this
    placement; used by the policy to compute the immediate reward for
    beam-search scoring.
    """
    g = grid.copy()
    for dr, dc in piece.cells:
        g[row + dr, col + dc] = 1
    full_rows = np.where(g.all(axis=1))[0]
    full_cols = np.where(g.all(axis=0))[0]
    lines_cleared = len(full_rows) + len(full_cols)
    if lines_cleared:
        g[full_rows, :] = 0
        g[:, full_cols] = 0
    return g, lines_cleared


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
#   (first_action:      tuple[int,int,int],
#    current_grid:      np.ndarray,
#    remaining_pieces:  list[Piece],
#    accum_reward:      float       # Σ γ^k · r_k for placements done so far
#    depth:             int)        # # of placements done so far (= γ exponent for next step's leaf)
_BeamEntry = tuple


def _leaf_score(accum_reward: float, depth: int, vstar: float) -> float:
    """score = accum_reward + γ^depth · V*(leaf).

    `accum_reward` already has γ-discounting baked in (each r_k was added
    as γ^k · r_k), so the V* term gets γ^depth on top to keep horizons
    consistent.
    """
    return accum_reward + (param.GAMMA ** depth) * vstar


def _terminal_score(accum_reward: float) -> float:
    """Dead-end leaf: V*=0 (terminal), keep whatever reward we already earned."""
    return accum_reward


def _sample_action(
    candidates: list[tuple[tuple[int, int, int], float]],
    temperature: float,
    top_m: int,
) -> tuple[int, int, int]:
    """Pick an action from `(first_action, score)` candidates.

    - `temperature == 0.0`: argmax (deterministic, eval-safe).
    - `temperature > 0`: dedupe by first_action keeping its best score, take
      the top-M, then sample one with probability softmax(score / τ).

    The dedupe matters because the same `first_action` can appear in many
    candidates (different queue orderings, different downstream placements);
    we want each distinct *first move* to compete on its best continuation.
    """
    if not candidates:
        raise RuntimeError("_sample_action called with no candidates")

    if temperature <= 0.0:
        best_action, _ = max(candidates, key=lambda x: x[1])
        return best_action

    best_by_action: dict[tuple[int, int, int], float] = {}
    for action, score in candidates:
        if action not in best_by_action or score > best_by_action[action]:
            best_by_action[action] = score

    ranked = sorted(best_by_action.items(), key=lambda kv: kv[1], reverse=True)
    pool = ranked[: max(1, top_m)]
    if len(pool) == 1:
        return pool[0][0]

    scores = [s for _, s in pool]
    s_max = max(scores)
    weights = [math.exp((s - s_max) / temperature) for s in scores]
    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for (action, _), w in zip(pool, weights):
        acc += w
        if r <= acc:
            return action
    return pool[-1][0]


def select_action(
    env: BlockBlastEnv,
    net: Optional[ValueNet],
    epsilon: float = 0.0,
    device: str | None = None,
    temperature: float = 0.0,
    top_m: int | None = None,
) -> tuple[int, int, int]:
    """Choose a (slot, row, col) action via beam-search 3-piece lookahead.

    Strategy:
      - With probability `epsilon` or if `net` is None: uniform random from
        the 1-step legal action set.
      - Otherwise: beam search over all distinct orderings of the queue.
        At each of the 3 depths, candidates are scored as V*(board, remaining)
        and only the top BEAM_WIDTH are expanded at the next depth.
        Dead-end beams (no legal position for the next piece) score V=0.
      - Final selection: if `temperature == 0` (default), pick the argmax over
        all leaf candidates (deterministic; eval-safe).  If `temperature > 0`,
        dedupe leaf candidates by first_action and sample one of the top-`top_m`
        with probability proportional to `exp(score / temperature)` — used
        during data-collection rounds to inject state diversity into the buffer.
    """
    actions = env.legal_actions()
    if not actions:
        raise RuntimeError("select_action called on a terminal state (no legal actions)")

    if net is None or random.random() < epsilon:
        return random.choice(actions)

    if top_m is None:
        top_m = param.SIM_EXPLORE_TOP_M

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

    gamma = param.GAMMA

    # ------------------------------------------------------------------
    # Depth 0 -> 1: expand the first piece of every distinct ordering.
    # Each candidate carries accum_reward = r_0 (γ^0 · r_0).
    # ------------------------------------------------------------------
    d1_cands: list[_BeamEntry] = []
    for ordering in distinct_orderings:
        p0 = queue[ordering[0]]
        remaining = [queue[ordering[i]] for i in range(1, n_pieces)]
        for r, c in _legal_positions(grid, p0):
            new_grid, lines = _place_and_clear(grid, p0, r, c)
            r0 = step_reward(len(p0.cells), lines)
            d1_cands.append(
                ((ordering[0], r, c), new_grid, remaining, r0, 1)
            )

    if not d1_cands:
        return random.choice(actions)

    # Score depth-1 candidates as r_0 + γ · V*(g1) so the top-K cut is
    # made on the right objective (not on V*(g1) alone).
    d1_vstar = _batch_score(
        [c[1] for c in d1_cands],
        [c[2] for c in d1_cands],
        net, net_device, chunk,
    )
    d1_scores = [
        _leaf_score(c[3], c[4], v) for c, v in zip(d1_cands, d1_vstar)
    ]
    beams: list[_BeamEntry] = _top_k(d1_cands, d1_scores, width)

    # Collect ALL leaf-score candidates across depths.  Each entry is
    # (first_action, score); the same first_action can appear multiple times
    # (e.g. via different queue orderings or downstream placements) and the
    # sampler dedupes by keeping each first_action's best score.
    leaf_candidates: list[tuple[tuple[int, int, int], float]] = []

    # ------------------------------------------------------------------
    # Depth 1 -> 2: expand the second piece on surviving beams.
    # ------------------------------------------------------------------
    d2_cands: list[_BeamEntry] = []
    for first_action, g1, remaining, accum1, _depth1 in beams:
        if not remaining:
            # Queue had only 1 piece — this beam is already a leaf at depth 1.
            vstar = _batch_score([g1], [[]], net, net_device, chunk)[0]
            leaf_candidates.append((first_action, _leaf_score(accum1, 1, vstar)))
            continue
        p1 = remaining[0]
        positions = _legal_positions(g1, p1)
        if not positions:
            # Dead-end at depth 2: keep accum_reward, no future value.
            leaf_candidates.append((first_action, _terminal_score(accum1)))
            continue
        rem2 = remaining[1:]
        for r, c in positions:
            new_grid, lines = _place_and_clear(g1, p1, r, c)
            r1 = step_reward(len(p1.cells), lines)
            accum2 = accum1 + (gamma ** 1) * r1
            d2_cands.append((first_action, new_grid, rem2, accum2, 2))

    if not d2_cands:
        return _sample_action(leaf_candidates, temperature, top_m)

    d2_vstar = _batch_score(
        [c[1] for c in d2_cands],
        [c[2] for c in d2_cands],
        net, net_device, chunk,
    )
    d2_scores = [
        _leaf_score(c[3], c[4], v) for c, v in zip(d2_cands, d2_vstar)
    ]
    beams = _top_k(d2_cands, d2_scores, width)

    # ------------------------------------------------------------------
    # Depth 2 -> 3 (final): expand the third piece, collect all leaves.
    # Each candidate is (first_action, g3, accum3) — empty queue at leaf.
    # ------------------------------------------------------------------
    d3_cands: list[tuple[tuple[int, int, int], np.ndarray, float]] = []
    for first_action, g2, remaining, accum2, _depth2 in beams:
        if not remaining:
            # Queue had only 2 pieces — leaf at depth 2.
            vstar = _batch_score([g2], [[]], net, net_device, chunk)[0]
            leaf_candidates.append((first_action, _leaf_score(accum2, 2, vstar)))
            continue
        p2 = remaining[0]
        positions = _legal_positions(g2, p2)
        if not positions:
            leaf_candidates.append((first_action, _terminal_score(accum2)))
            continue
        for r, c in positions:
            new_grid, lines = _place_and_clear(g2, p2, r, c)
            r2 = step_reward(len(p2.cells), lines)
            accum3 = accum2 + (gamma ** 2) * r2
            d3_cands.append((first_action, new_grid, accum3))

    if d3_cands:
        d3_vstar = _batch_score(
            [c[1] for c in d3_cands],
            [[] for _ in d3_cands],
            net, net_device, chunk,
        )
        for (first_action, _, accum3), v in zip(d3_cands, d3_vstar):
            leaf_candidates.append((first_action, _leaf_score(accum3, 3, v)))

    return _sample_action(leaf_candidates, temperature, top_m)
