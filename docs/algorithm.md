# Algorithm

[← back to README](../README.md)

A CNN state-value network `v(s)` is trained via Monte Carlo returns with
**potential-based reward shaping** (Ng, Harada, Russell 1999) and **D4
symmetry augmentation**. The agent plays by greedily picking the action whose
afterstate has the highest predicted value (with the shaping potential added
back at decision time so the optimal policy is unchanged).

- [State encoding](#state-encoding)
- [Value network](#value-network)
- [Monte Carlo training pipeline](#monte-carlo-training-pipeline)
- [Reward shaping (potential-based)](#reward-shaping-potential-based)
- [Data augmentation (D4 symmetries)](#data-augmentation-d4-symmetries)
- [Optimizer state persistence](#optimizer-state-persistence)
- [Champion / challenger checkpointing](#champion--challenger-checkpointing)

## State encoding

`encoder.encode_state(board, queue)` returns a `(4, 8, 8)` float32 tensor:

| Channel | Content |
|---------|---------|
| 0 | Board occupancy (0 = empty, 1 = filled) |
| 1 | Queued piece #1 rasterised into top-left of 8×8 plane |
| 2 | Queued piece #2 rasterised into top-left of 8×8 plane |
| 3 | Queued piece #3 rasterised into top-left of 8×8 plane |

## Value network

```
Conv2d(4 → C, 3×3, pad=1) → ReLU
Conv2d(C → C, 3×3, pad=1) → ReLU
Conv2d(C → C, 3×3, pad=1) → ReLU
Flatten → Linear(C×64 → H) → ReLU → Linear(H → 1)
```

`C = CNN_CHANNELS`, `H = HIDDEN_SIZE` (see [hyperparameters.md](hyperparameters.md)).

## Monte Carlo training pipeline

```
simulate.py                       train.py
    │                                 │
    ├─ load checkpoint if exists      ├─ glob simulations/*.json
    ├─ for each episode:              ├─ episode-level train/test split
    │    ε-greedy policy via v(s')    ├─ compute G_t for each step
    │    record (board, queue,        ├─ for NUM_EPOCHS:
    │            action, reward)      │    minibatch MSE v(s) vs G_t
    └─ write ep_*.json                └─ save best-by-test-loss checkpoint
```

The policy uses **3-piece beam-search lookahead** — the full queue is planned
in one shot using beam search to keep cost tractable:

```
Depth 0->1  place piece A at all legal positions (~40),
            score V*(board_A, queue=[B,C]) for each,
            keep top BEAM_WIDTH candidates.

Depth 1->2  expand each beam with piece B (~40 positions each),
            score V*(board_AB, queue=[C]),
            keep top BEAM_WIDTH.

Depth 2->3  expand each beam with piece C, score V*(board_ABC, queue=[]),
            return first_action of the global argmax.
```

Intermediate states are scored with `V*(board) = V_F(board) + Phi(board)`,
passing the remaining unplaced pieces as queue context so the net sees the
same input distribution it was trained on. Dead-end beams (no legal position
for the next piece) are scored as `V = 0`. All distinct piece orderings are
tried and the best `first_action` across all is returned.

Total network forward calls per move: ~40 (depth 1) + K×40 (depth 2) +
K×40 (depth 3) ≈ 440 for `BEAM_WIDTH = 5`, versus up to 750 k for exhaustive
enumeration.

MC returns `G_t = Σ_{k≥t} γ^(k−t) · r_k` align the value function directly
with "survive as long as possible", which is the stated objective.

Iterating simulate → train improves data quality over rounds:

```
Round 1: simulate (random) → train → checkpoint v1
Round 2: simulate (greedy v1 + ε) → train → checkpoint v2
Round 3: simulate (greedy v2 + ε) → train → checkpoint v3
  ...
```

## Reward shaping (potential-based)

Block Blast has a hard exploration problem: line clears (especially 3+ line
combos worth 50–250 points) are rare under any near-random policy. Without
shaping the value net almost never sees a state that led to a multi-clear,
so it cannot learn to value set-ups that lead to one. The agent plateaus on
"just place pieces until you can't".

We address this with **potential-based reward shaping** (Ng, Harada, Russell
1999). Define a potential function over states `Phi: S → ℝ` and add a shaped
reward `F(s, s') = γ·Phi(s') − Phi(s)` at every transition. The key
theoretical guarantee is that this **preserves the optimal policy** — every
optimal action under the original reward is still optimal under the shaped
reward, regardless of how `Phi` is chosen.

**The potential function** ([`blockblaster/game/potential.py`](../blockblaster/game/potential.py))
has three complementary terms:

1. **Row/column fill (quadratic):** rewards near-complete rows / columns so
   7/8 fill is worth far more than 5/8, pushing the agent to set up line
   clears before it has ever executed one.

2. **Transition penalty (subtracted):** counts adjacent cell pairs that flip
   between filled and empty, summed over all rows and all columns. Lower =
   better. A solid block `[1,1,1,1,0,0,0,0]` has 1 transition; a checkerboard
   `[1,0,1,0,1,0,1,0]` has 7. Penalises fragmented, interleaved boards where
   pieces are unlikely to complete lines. Computed in two numpy lines — no
   loop.

3. **Piece fittability:** for each piece type `p`, contributes
   `|p| × num_legal_placements(p, board)`. A board where the 3×3 square has
   no legal placement contributes 0 for those 9 cells, directly penalising
   the state. Computed efficiently via numpy boolean slice intersection — no
   Python loop over board positions.

```
Phi(s) = POTENTIAL_COEFF   · ( Σ_rows row_fill² + Σ_cols col_fill² )
       - TRANSITIONS_COEFF · ( Σ_rows transitions(row) + Σ_cols transitions(col) )
       + FITTABILITY_COEFF · Σ_{p ∈ PIECES} |p| · num_legal_placements(p, board)
```

Examples (with `POTENTIAL_COEFF = 0.07`, `TRANSITIONS_COEFF = 0.1`, `FITTABILITY_COEFF = 0.03`):

| Board state | Fill | Transitions penalty | Fittability | Phi |
|-------------|------|---------------------|-------------|-----|
| Empty | 0.0 | −0.0 (0 transitions) | ~150 | ~150 |
| Half board: solid 4×8 block | ~14 | −0.9 (9 transitions) | ~75 | ~88 |
| Half board: checkerboard fill | ~14 | −5.6 (56 transitions) | ~20 | ~28 |
| Completely full board | ~72 | −0.0 (0 transitions) | 0.0 | ~72 |
| Terminal state | 0 (by convention) | 0 | 0 | 0 |

**Where the shaping is applied.** Per-step shaped rewards along a Monte
Carlo trajectory telescope into a single correction at the start state:

```
G_t^F  =  Σ γ^(k−t) (r_k + F_k)
       =  G_t  +  γ^(T−t) Phi(s_T) − Phi(s_t)
       =  G_t  −  Phi(s_t)                          (Phi(terminal) ≡ 0)
```

So instead of changing the env's per-step reward, we apply the shaping
**once** in the dataset by subtracting `Phi(s_t)` from each Monte Carlo
target. The value net learns `V_F(s) = V*(s) − Phi(s)`.

**Where the shaping is undone.** At action-selection time the policy adds
`Phi(s')` back to each candidate afterstate so the argmax recovers `V*(s')`:

```python
# blockblaster/agent/policy.py
values   = net.predict(afterstates)     # = V_F(s') = V*(s') − Phi(s')
scores   = values + phi_vec             # = V*(s')
best_idx = scores.argmax()
```

This preserves the original optimal policy while giving the value net a much
denser training signal: states with near-complete rows / columns, low
fragmentation, and many legal placements for all piece types have distinctly
higher targets, so the value gradient pushes toward line-clear set-ups,
consolidated board geometry, and avoiding piece-deadlock situations.

Episode JSONs store the **unshaped** game reward, so reported `total_score`
and the round summary printed by `run_loop.py` reflect the real game score —
not the shaped one. Shaping only affects training targets and action
selection.

## Data augmentation (D4 symmetries)

An 8 × 8 board has 8 dihedral symmetries (4 rotations × 2 reflections — the
group `D4`). The game's value function is invariant under this group, so
every `(state, target)` pair from a real episode generates 7 additional
training pairs with identical target. `EpisodeDataset` enumerates these when
`USE_DIHEDRAL_AUG=True`, expanding the training set ~8×:

```python
for k in range(4):
    rot = torch.rot90(tensor, k, dims=(-2, -1))
    variants.append(rot)
    variants.append(torch.flip(rot, dims=(-1,)))
```

Augmentation is applied only to the training split (not the test split) so
test loss remains an honest unbiased estimate.

## Optimizer state persistence

Checkpoints persist the Adam optimizer state (`exp_avg`, `exp_avg_sq`,
`step`) alongside the model weights, so accumulated curvature information
survives across simulate → train rounds and the optimizer doesn't restart
from zero momentum every iteration.

## Champion / challenger checkpointing

Two checkpoints are maintained in parallel:

| File | Role |
|------|------|
| `checkpoints/value_net.pt`      | **Challenger** — the latest weights produced by `train()`; updated every round. |
| `checkpoints/best_value_net.pt` | **Champion** — the stable policy used for data collection; only ever advances. |

`run_loop.py` orchestrates a champion/challenger evaluation:

```
Normal round              (round % EVAL_INTERVAL != 0)
    sim policy = champion (BEST_CHECKPOINT_PATH)
    → episodes collected with the stable policy; best_median_ever is NOT
       updated even if the round's sample median happens to exceed it
       (avoids drift from sampling noise without an actual policy change).

Eval round                (round % EVAL_INTERVAL == 0)
    sim policy = challenger (CHECKPOINT_PATH, the freshly trained weights)
    if median(challenger) > best_median_ever:
        promote: copy CHECKPOINT_PATH → BEST_CHECKPOINT_PATH
        best_median_ever = median(challenger)
    else:
        keep current champion
```

We compare on **median**, not mean: a single unusually-long lucky episode
can drag the mean far above typical play, and we don't want one tail event
tipping a promotion decision. Median is the typical-game score.

Early-round bootstrap: until BEST exists (typically round 1 cold-start →
random policy, then a round or two of CHECKPOINT-only play), the resolver
falls back to CHECKPOINT for normal rounds and `median > -inf` is treated
as the first promotion that seeds BEST.

Why this design — without it, two failure modes are easy to hit:

1. **Sim-from-latest only** — training instabilities propagate into the
   data-collection policy and `median` can collapse from e.g. 320 back down
   to 60 over a few hundred rounds, even though earlier weights were good.
2. **Sim-from-best only** — once BEST is set, the simulator never tries the
   actively trained CHECKPOINT, so BEST is effectively frozen and the loop
   stops improving.

The eval round breaks both: most rounds use the stable champion so data
quality doesn't regress, and every `EVAL_INTERVAL`th round gives the
challenger a fair head-to-head shot at the title.
