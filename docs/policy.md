# Beam-search policy

How `blockblaster.agent.policy.select_action` picks a move, and why it
scores beam candidates the way it does.

## TL;DR

The policy is a 3-piece beam search over the queue.  Each candidate
sequence (a choice of where to place each of the 3 queued pieces) is
scored as the **true discounted return** it would produce:

\[
\text{score} \;=\; r_0 \;+\; \gamma\, r_1 \;+\; \gamma^2\, r_2 \;+\; \gamma^3\, V^*(s_3)
\]

where:

- \(r_k\) is the immediate reward earned by placing piece \(k\) (cells
  placed + line-clear bonuses, per `blockblaster.game.scoring`).
- \(s_3\) is the board after all 3 placements.
- \(V^*(s)\) is the value of state \(s\), reconstructed from the trained
  net's shaped output: \(V^*(s) = V_F(s) + \Phi(s)\).
- \(\gamma = \texttt{param.GAMMA}\).

The first move of the best-scoring sequence is returned.

## How the beam runs

The queue has `param.QUEUE_SIZE` pieces (default 3).  For each distinct
permutation of the queue (dedup'd by piece id), the search explores:

| Depth | Action | Candidates kept |
|---|---|---|
| 0 → 1 | Place piece A at every legal position | top `BEAM_WIDTH` by `r_0 + γ V*(s_1)` |
| 1 → 2 | Expand each beam by placing piece B at every legal position | top `BEAM_WIDTH` by `r_0 + γ r_1 + γ² V*(s_2)` |
| 2 → 3 | Expand each beam by placing piece C at every legal position | global argmax by `r_0 + γ r_1 + γ² r_2 + γ³ V*(s_3)` |

At each depth, the value net is called once on a batch of all surviving
candidate states (chunked by `LOOKAHEAD_MAX_BATCH` to bound VRAM).  The
remaining-queue context passed to the encoder for intermediate states is
the pieces *not yet placed* in that beam — consistent with how the net
was trained.

### Dead-ends and short queues

- A beam whose next piece has no legal placement is scored as its
  **accumulated reward so far**, with \(V^*=0\) (terminal state — no
  future value to bootstrap on, but the rewards we already earned are
  still real).
- A beam that runs out of pieces before depth 3 (e.g. queue had only 1
  or 2 pieces) is scored at its actual leaf depth: `accum_reward +
  γ^depth · V*(leaf)`.

### Distinct orderings

The queue's 3 pieces have `3! = 6` orderings, but duplicate piece ids
collapse them (e.g. `[L, L, S]` has only 3 distinct orderings).  Each
distinct ordering seeds its own depth-1 candidates; the global argmax
across all orderings picks the first action returned.

### Exploration: softmax over final leaves (preferred) and ε-greedy (fallback)

There are two exploration mechanisms, used in different situations:

- **Softmax sampling over the beam's top-M final leaves**
  (`param.SIM_TEMPERATURE`, `param.SIM_EXPLORE_TOP_M`). When `τ > 0` and a
  net is available, `select_action` collects all final-depth leaf
  candidates, dedupes by `first_action` (keeping each first move's best
  continuation), keeps the top-M by score, and samples one with
  probability `softmax(score / τ)`. This is the recommended exploration
  knob during **data-collection** rounds: the policy only ever picks moves
  the search already rated highly, so we get trajectory diversity without
  ever committing obviously bad moves. `τ = 0` reproduces the deterministic
  argmax. Eval rounds always pass `τ = 0` so paired champion-vs-challenger
  comparisons are noise-free.

- **ε-greedy uniform random** (`param.SIM_EPSILON`). With probability
  `epsilon` (or unconditionally when `net is None`), `select_action`
  returns a uniform random legal action from `env.legal_actions()` instead
  of running the beam. Kept primarily for the cold-start / no-checkpoint
  case; for net-driven exploration, prefer `SIM_TEMPERATURE` — uniform
  random placements on a cramped board are near-suicide and corrupt the
  dataset with truncated episodes.

See [`sim-configs.md`](sim-configs.md) for recommended settings.

## Why score sequences as full returns, not just `V*(leaf)`

An earlier version of this policy scored each candidate sequence as
`V*(s_3) + Φ(s_3)` alone — i.e. it ignored the immediate rewards
earned during the 3 placements and asked the value net "how good does
the board look after all 3 placements?"

That breaks badly on cramped boards.  Concrete example:

- **Sequence A:** placement 1 clears 2 lines (`r_0 = 8 + 20 + 20 = 48`),
  placements 2 and 3 are normal fills (`r_1 = r_2 ≈ 4`).  Board at
  \(s_3\): newly cleared then partially refilled.
- **Sequence B:** no clears in any of the 3 placements (`r_0 = r_1 =
  r_2 ≈ 4`).  Board at \(s_3\): tighter packed, possibly looks
  marginally better to `V + Φ`.

If `V*(s_3^B) + Φ(s_3^B)` is even slightly higher than
`V*(s_3^A) + Φ(s_3^A)`, the old policy picks B and **declines the
+48 line-clear**.  Late-game this is fatal — the next piece arrives,
the un-cleared board can't fit it, and the game ends.

The trained network expects rewards to be counted in the return:
\(G_t = \sum_{k} \gamma^k r_{t+k}\) is the MC target.  At inference,
ignoring \(r_0, r_1, r_2\) is a consistency bug.  Scoring sequences as
the full discounted return makes the lookahead and the value net agree
on what they're optimising.

## Where this is implemented

- `_place_and_clear(grid, piece, row, col) -> (new_grid, lines_cleared)`
  applies a placement and reports how many rows + columns cleared.  The
  caller computes the reward from this via
  `blockblaster.game.scoring.step_reward(len(piece.cells), lines)`.
- Beam entries carry `(first_action, current_grid, remaining_pieces,
  accum_reward, depth)`.  `accum_reward` is already γ-discounted (each
  `r_k` is added in as `γ^k · r_k`); leaf scoring is `accum_reward +
  γ^depth · V*(leaf)`.
- `_leaf_score` centralises the non-terminal leaf math so the three
  depth blocks stay readable.  Dead-end (game-over) leaves take their
  raw `accum_reward` (V*=0 by definition) and are tagged
  `terminal=True` in the leaf candidate tuple; `_sample_action`
  filters them out whenever any non-terminal leaf exists, so a
  high-immediate-reward suicide path can never beat a survivable one
  on the raw argmax.

## Parameter knobs that affect the policy

From `param.py`:

| Param | Effect |
|---|---|
| `BEAM_WIDTH` | Beams kept per depth.  Quality scales sublinearly; 32–64 captures almost all of the gain past 10.  See [`sim-configs.md`](sim-configs.md). |
| `LOOKAHEAD_DEPTH` | Currently equals `QUEUE_SIZE` (3).  Going deeper requires sampling unknown future pieces — not implemented. |
| `LOOKAHEAD_MAX_BATCH` | VRAM bound on per-depth net forwards.  Larger = fewer kernel launches when the beam is wide. |
| `GAMMA` | Discount factor for both training targets and beam scoring.  These two must agree — don't tune one without the other. |
| `POTENTIAL_COEFF`, `TRANSITIONS_COEFF`, `FITTABILITY_COEFF` | Shape \(\Phi(s)\), which is added to `V_F(s)` to recover `V*(s)`.  Heavier shaping makes the policy more conservative about board structure. |

## Future improvements (not implemented)

1. **Dihedral-averaged value.**  Average `V*` over the 8 symmetries of
   the board (rot×flip).  The net is already trained with
   `USE_DIHEDRAL_AUG=True`; evaluating one orientation at inference is
   leaving variance reduction on the table.

2. **MC rollouts at beam leaves.**  Replace `V*(leaf)` with a short
   greedy playout: keep stepping with the policy for \(K\) more pieces,
   accumulate `Σ γ^k r_k`, then bootstrap with `γ^K · V*(s_K)`.
   Anchors the bootstrap in many real placements instead of a single
   value estimate.  Big late-game improvement, modest code change.

3. **MCTS / PUCT** with `V*` as the leaf evaluator and a uniform (or
   learned) policy prior.  AlphaZero-flavoured self-play — highest
   quality, biggest lift.  Also enables storing `π_MCTS` per step as a
   policy-improvement target, which is a much stronger training signal
   than the single greedy action.
