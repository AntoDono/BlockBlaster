"""Shared 3-piece lookahead search for the training policy and assist advisor.

Both the training agent (:mod:`blockblaster.agent.policy`) and the live assist
advisor (:mod:`blockblaster.assist.advisor`) score a candidate *first move*
the same way:

    score = r_0 + γ·r_1 + γ²·r_2 + γ³·V*(s_3)

where ``r_k`` is the immediate reward (cells placed + line-clear bonuses) for
the ``k``-th simulated placement and ``V*(s) = V_F(s) + Φ(s)`` is the
shaping-corrected value-net estimate.  Each placement is fully simulated
(rows/columns that fill are cleared) before the next piece is placed.

Selection has two tiers:

1. **Feasibility first** — a *non-terminal* leaf is one where every queued
   piece was placed.  If any first move has at least one non-terminal leaf,
   only those first moves are considered.
2. **Reward tie-break** — within the chosen pool, pick by score (argmax for
   ``temperature == 0``, softmax over the top-``M`` otherwise).

The training policy expands a beam of ``BEAM_WIDTH`` per depth; the assist
advisor expands exhaustively (``beam_width=None``) so it never prunes the
sole feasible first move.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import permutations
from typing import Optional

import numpy as np
import torch

import param
from blockblaster.game.board import Board, legal_positions_grid
from blockblaster.game.pieces import Piece
from blockblaster.game.potential import board_potential
from blockblaster.game.scoring import step_reward
from blockblaster.model.encoder import encode_state
from blockblaster.model.value_net import ValueNet


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def place_and_clear(
    grid: np.ndarray, piece: Piece, row: int, col: int,
) -> tuple[np.ndarray, int]:
    """Place ``piece`` at ``(row, col)``; clear any newly-full rows/columns.

    Returns ``(new_grid, lines_cleared)`` where ``lines_cleared`` is the total
    number of full rows + full columns cleared by this placement (matching the
    in-game scoring convention).
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
# Net scoring
# ---------------------------------------------------------------------------

def batch_vstar(
    grids: list[np.ndarray],
    queues: list[list[Piece]],
    net: ValueNet,
    net_device: torch.device,
    chunk_size: int,
) -> list[float]:
    """Return ``V*(grid, queue) = V_F(grid) + Φ(grid)`` for each pair.

    Encodes boards in chunks of ``chunk_size`` to bound GPU memory.  The queue
    is passed to the encoder so intermediate states use the correct
    remaining-piece context (consistent with how the net was trained).
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


def leaf_score(accum_reward: float, depth: int, vstar: float) -> float:
    """``score = accum_reward + γ^depth · V*(leaf)``.

    ``accum_reward`` already has γ-discounting baked in (each ``r_k`` was
    added as ``γ^k · r_k``), so the ``V*`` term gets ``γ^depth`` on top to
    keep horizons consistent.
    """
    return accum_reward + (param.GAMMA ** depth) * vstar


def _top_k(candidates: list, scores: list[float], k: Optional[int]) -> list:
    """Keep the top-``k`` candidates by descending score, or all if ``k`` is None."""
    if k is None or len(candidates) <= k:
        return list(candidates)
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
    return [candidates[i] for i in order[:k]]


# ---------------------------------------------------------------------------
# Leaf enumeration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannedStep:
    """One placement in a lookahead sequence."""

    slot: int
    row: int
    col: int
    piece: Piece


# A leaf candidate: ``(path, score, terminal)``.
#   - ``path`` is the ordered tuple of :class:`PlannedStep` placements taken
#     to reach this leaf.  ``path[0]`` is the first move (what the search
#     ultimately selects); deeper entries describe the planned follow-ups.
#   - ``terminal=False``: every planned piece placed successfully; ``score``
#     is the full discounted return including ``γ^depth · V*(leaf)``.
#   - ``terminal=True``: a dead-end was hit (the next planned piece had no
#     legal position); ``score`` is the discounted reward accumulated up to
#     the dead-end (``V* = 0``).  ``path`` covers only the placements that
#     succeeded before the dead-end.
LeafCandidate = tuple[tuple["PlannedStep", ...], float, bool]


@dataclass(frozen=True)
class FirstMoveResult:
    """Top-level outcome of :func:`search_first_move`.

    ``plan`` is the full lookahead sequence that produced ``score`` —
    ``plan[0]`` matches ``(slot, row, col, piece)`` on this result; later
    entries describe the planned follow-up placements.  When ``terminal``
    is true the plan stops at the last successful placement before the
    dead-end.
    """

    slot: int
    row: int
    col: int
    piece: Piece
    score: float
    terminal: bool
    plan: tuple[PlannedStep, ...]


# Internal beam-entry shape:
#   (path, current_grid, remaining_pieces, accum_reward, depth)
# ``path[0]`` is the first move (what the search ultimately commits to).
_BeamEntry = tuple


def enumerate_leaves(
    grid: np.ndarray,
    tray: list[tuple[int, Piece]],
    net: ValueNet,
    *,
    beam_width: Optional[int] = None,
    keep_all_first_moves: bool = False,
    chunk_size: Optional[int] = None,
) -> list[LeafCandidate]:
    """Enumerate scored leaves for every distinct ordering of ``tray``.

    ``tray`` is a list of ``(slot, piece)`` pairs — the slot is the
    caller-facing index (queue position for training, tray index 0/1/2 for
    assist) and is what each leaf's ``first_action`` refers to.

    ``beam_width=K`` keeps the top-``K`` candidates at each depth (the
    training policy's `param.BEAM_WIDTH`); ``None`` runs an unbounded search.
    ``keep_all_first_moves=True`` forces *every* depth-1 candidate to expand
    (deeper depths still use ``beam_width``); the assist advisor passes
    this so feasibility is guaranteed for every possible first move while
    keeping deeper expansion tractable.

    For partial trays (fewer than ``QUEUE_SIZE`` pieces) the search depth
    adapts naturally.
    """
    if not tray:
        return []
    if chunk_size is None:
        chunk_size = param.LOOKAHEAD_MAX_BATCH

    net_device = next(net.parameters()).device
    gamma = param.GAMMA
    n_pieces = len(tray)
    slots = [s for s, _ in tray]
    pieces = [p for _, p in tray]

    # Distinct orderings deduped by piece-id tuple — two trays with the same
    # piece set in different slot order produce the same score landscape
    # for the planning side; we still dispatch the original slot index in
    # ``first_action`` so the slot mapping is preserved per ordering.
    seen: set[tuple[int, ...]] = set()
    distinct_orderings: list[tuple[int, ...]] = []
    for perm in permutations(range(n_pieces)):
        pid_key = tuple(pieces[i].piece_id for i in perm)
        if pid_key not in seen:
            seen.add(pid_key)
            distinct_orderings.append(perm)

    # Depth 0 -> 1: expand the first piece of every distinct ordering.
    d1_cands: list[_BeamEntry] = []
    for ordering in distinct_orderings:
        first_idx = ordering[0]
        p0 = pieces[first_idx]
        remaining_idx = [ordering[i] for i in range(1, n_pieces)]
        remaining = [pieces[i] for i in remaining_idx]
        remaining_slots = [slots[i] for i in remaining_idx]
        for r, c in legal_positions_grid(grid, p0):
            new_grid, lines = place_and_clear(grid, p0, r, c)
            r0 = step_reward(len(p0.cells), lines)
            step0 = PlannedStep(slot=slots[first_idx], row=r, col=c, piece=p0)
            d1_cands.append(
                ((step0,), new_grid, list(zip(remaining_slots, remaining)), r0, 1)
            )

    if not d1_cands:
        return []

    def _queues(entries):
        # ``remaining`` is now a list of (slot, piece); the net only cares
        # about the piece list for state encoding.
        return [[p for _, p in c[2]] for c in entries]

    d1_vstar = batch_vstar(
        [c[1] for c in d1_cands],
        _queues(d1_cands),
        net, net_device, chunk_size,
    )
    d1_scores = [leaf_score(c[3], c[4], v) for c, v in zip(d1_cands, d1_vstar)]
    d1_cut = None if keep_all_first_moves else beam_width
    beams: list[_BeamEntry] = _top_k(d1_cands, d1_scores, d1_cut)

    leaves: list[LeafCandidate] = []

    # Depth 1 -> 2.  Orderings are already enumerated at depth 1, so we
    # only advance the ordering's next piece here (``remaining[0]``); a
    # different next-piece choice is captured by another permutation.
    d2_cands: list[_BeamEntry] = []
    for path, g1, remaining, accum1, _depth1 in beams:
        if not remaining:
            vstar = batch_vstar([g1], [[]], net, net_device, chunk_size)[0]
            leaves.append((path, leaf_score(accum1, 1, vstar), False))
            continue
        slot_p1, p1 = remaining[0]
        positions = legal_positions_grid(g1, p1)
        if not positions:
            leaves.append((path, accum1, True))
            continue
        rem2 = remaining[1:]
        for r, c in positions:
            new_grid, lines = place_and_clear(g1, p1, r, c)
            r1 = step_reward(len(p1.cells), lines)
            accum2 = accum1 + (gamma ** 1) * r1
            step1 = PlannedStep(slot=slot_p1, row=r, col=c, piece=p1)
            d2_cands.append((path + (step1,), new_grid, rem2, accum2, 2))

    if not d2_cands:
        return leaves

    d2_vstar = batch_vstar(
        [c[1] for c in d2_cands],
        _queues(d2_cands),
        net, net_device, chunk_size,
    )
    d2_scores = [leaf_score(c[3], c[4], v) for c, v in zip(d2_cands, d2_vstar)]
    beams = _top_k(d2_cands, d2_scores, beam_width)

    # Depth 2 -> 3 (final).
    d3_cands: list[tuple[tuple[PlannedStep, ...], np.ndarray, float]] = []
    for path, g2, remaining, accum2, _depth2 in beams:
        if not remaining:
            vstar = batch_vstar([g2], [[]], net, net_device, chunk_size)[0]
            leaves.append((path, leaf_score(accum2, 2, vstar), False))
            continue
        slot_p2, p2 = remaining[0]
        positions = legal_positions_grid(g2, p2)
        if not positions:
            leaves.append((path, accum2, True))
            continue
        for r, c in positions:
            new_grid, lines = place_and_clear(g2, p2, r, c)
            r2 = step_reward(len(p2.cells), lines)
            accum3 = accum2 + (gamma ** 2) * r2
            step2 = PlannedStep(slot=slot_p2, row=r, col=c, piece=p2)
            d3_cands.append((path + (step2,), new_grid, accum3))

    if d3_cands:
        d3_vstar = batch_vstar(
            [c[1] for c in d3_cands],
            [[] for _ in d3_cands],
            net, net_device, chunk_size,
        )
        for (path, _, accum3), v in zip(d3_cands, d3_vstar):
            leaves.append((path, leaf_score(accum3, 3, v), False))

    return leaves


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _partition_best_by_action(
    leaves: list[LeafCandidate],
) -> tuple[dict, dict]:
    """Return ``(best_safe, best_unsafe)`` mapping first-action → best ``(score, path)``."""
    best_safe: dict[tuple[int, int, int], tuple[float, tuple[PlannedStep, ...]]] = {}
    best_unsafe: dict[tuple[int, int, int], tuple[float, tuple[PlannedStep, ...]]] = {}
    for path, score, terminal in leaves:
        if not path:
            continue
        first = path[0]
        action = (first.slot, first.row, first.col)
        target = best_unsafe if terminal else best_safe
        if action not in target or score > target[action][0]:
            target[action] = (score, path)
    return best_safe, best_unsafe


def select_first_move(
    leaves: list[LeafCandidate],
    *,
    temperature: float = 0.0,
    top_m: int = 1,
) -> Optional[tuple[tuple[PlannedStep, ...], float, bool]]:
    """Pick a first move from enumerated leaves.

    Feasibility first: if any action has a non-terminal leaf, the unsafe
    actions are filtered out entirely.  If every action is terminal, ranking
    falls back to discounted step-reward tie-break among the unsafe pool.

    ``temperature == 0`` → argmax (deterministic).  ``temperature > 0`` →
    softmax-sample one of the top-``M`` actions.

    Returns ``(path, score, terminal_pool)`` for the winning leaf — ``path``
    is the full planned placement sequence (``path[0]`` is the chosen first
    move) and ``terminal_pool`` is ``True`` if the choice came from the
    unsafe fallback pool.
    """
    if not leaves:
        return None
    best_safe, best_unsafe = _partition_best_by_action(leaves)
    pool = best_safe if best_safe else best_unsafe
    terminal_pool = not best_safe
    if not pool:
        return None

    if temperature <= 0.0:
        _, (score, path) = max(pool.items(), key=lambda kv: kv[1][0])
        return path, score, terminal_pool

    ranked = sorted(pool.items(), key=lambda kv: kv[1][0], reverse=True)
    head = ranked[: max(1, top_m)]
    if len(head) == 1:
        _, (score, path) = head[0]
        return path, score, terminal_pool

    scores = [s for _, (s, _) in head]
    s_max = max(scores)
    weights = [math.exp((s - s_max) / temperature) for s in scores]
    total = sum(weights)
    r = random.random() * total
    acc = 0.0
    for (_, (score, path)), w in zip(head, weights):
        acc += w
        if r <= acc:
            return path, score, terminal_pool
    _, (score, path) = head[-1]
    return path, score, terminal_pool


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def search_first_move(
    grid: np.ndarray,
    tray: list[tuple[int, Piece]],
    net: ValueNet,
    *,
    beam_width: Optional[int] = None,
    keep_all_first_moves: bool = False,
    chunk_size: Optional[int] = None,
    temperature: float = 0.0,
    top_m: int = 1,
) -> Optional[FirstMoveResult]:
    """Enumerate leaves + select the best first move; ``None`` if no legal moves.

    Convenience wrapper used by the assist advisor.  Returns a
    :class:`FirstMoveResult` resolving the chosen slot back to its
    :class:`Piece` for caller convenience.
    """
    leaves = enumerate_leaves(
        grid, tray, net,
        beam_width=beam_width,
        keep_all_first_moves=keep_all_first_moves,
        chunk_size=chunk_size,
    )
    pick = select_first_move(leaves, temperature=temperature, top_m=top_m)
    if pick is None:
        return None
    path, score, terminal = pick
    if not path:
        return None
    first = path[0]
    return FirstMoveResult(
        slot=first.slot, row=first.row, col=first.col, piece=first.piece,
        score=score, terminal=terminal, plan=path,
    )
