# Algorithm

[← back to README](../README.md)

A CNN state-value network `v(s)` is trained via **n-step TD with a target
network**, using **potential-based reward shaping** (Ng, Harada, Russell 1999)
and **D4 symmetry augmentation**. The agent plays by beam-searching over the
3-piece queue and scoring leaves with the predicted value (shaping potential
added back at decision time so the optimal policy is unchanged). During data
collection, the final beam choice is **softmax-sampled** among the top-M
leaves so the replay buffer accumulates diverse trajectories instead of
collapsing onto a single deterministic policy.

- [State encoding](#state-encoding)
- [Value network](#value-network)
- [n-step TD training pipeline](#n-step-td-training-pipeline)
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

## n-step TD training pipeline

```
simulate.py                       train.py
    │                                 │
    ├─ load checkpoint if exists      ├─ glob simulations/*.json
    ├─ for each episode:              ├─ episode-level train/test split
    │    beam search + softmax       ├─ build per-step (s_t, s_{t+n}, Σγ^k r_{t+k}) samples
    │    sampling over top-M leaves  ├─ initialise target net V_target ← V_θ
    │    record (board, queue,        ├─ for NUM_EPOCHS:
    │            action, reward)      │    minibatch MSE v(s_t) vs target_F
    │                                 │    refresh V_target every TARGET_REFRESH_BATCHES
    └─ write ep_*.json                └─ save best-by-test-loss checkpoint
```

The per-state regression target is

```
target_F(s_t) = Σ_{k=0..n-1} γ^k r_{t+k}              ← real rewards from the episode
              + bootstrap · γ^n · ( V_target(s_{t+n}) + Φ(s_{t+n}) )
              - Φ(s_t)
```

where `bootstrap = 1` if the episode lasts at least `n` more steps after `t`,
else `0` (in which case the V_target term drops out and the contribution from
beyond `t+n-1` is treated as zero — equivalent to a pure MC return for the
tail of an episode). The trained net outputs the shaped value
`V_F(s) = V*(s) - Φ(s)` so beam scoring at inference can recover `V*(s)` by
adding `Φ(s)` back.

Why TD instead of full Monte Carlo: with pure MC the target for `s_t` is
`Σ γ^k r_{t+k}` taken to the end of the episode under whatever policy played
it. If the replay buffer is dominated by one policy (the current champion),
that target is `V^champion(s)` — a fixed function — and once the net has fit
it, additional training does nothing. n-step TD replaces the tail of the sum
with `V_target(s_{t+n})`, which **changes as the net improves**, so training
keeps making progress even when the buffer composition doesn't change. The
target network is a periodically-refreshed snapshot of the live net used for
the bootstrap; it stabilises training by keeping the regression target a
fixed function for a window of batches (see `TARGET_REFRESH_BATCHES`).

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

The discounted-return target (whether full MC or its n-step TD truncation)
aligns the value function directly with "survive as long as possible", which
is the stated objective.

Iterating simulate → train improves data quality over rounds. Most rounds
collect data with the **champion** (`best_value_net.pt`) using softmax
sampling over the top-M beam leaves (`SIM_TEMPERATURE > 0`) so successive
champion rounds explore different trajectories. Every `EVAL_INTERVAL` rounds,
the **challenger** (`value_net.pt`) plays a paired head-to-head against the
champion on a fixed set of seeds (`EVAL_SEEDS`) with `τ = 0` for noise-free
comparison; the challenger is promoted iff it wins the gate (see
[Champion / challenger checkpointing](#champion--challenger-checkpointing)).

```
Round 1: simulate (random)               → train → checkpoint v1
Round 2: simulate (champion + softmax)   → train → checkpoint v2
Round 3: simulate (champion + softmax)   → train → checkpoint v3
Round 4: eval round (paired v3 vs champ) → promote? → train → checkpoint v4
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

**Where the shaping is applied.** Per-step shaped rewards `F_k = γ·Φ(s_{k+1})
− Φ(s_k)` telescope along any trajectory of length `m`:

```
Σ_{k=0..m-1} γ^k F_k  =  γ^m Φ(s_m) − Φ(s_0)
```

For an n-step TD target this gives, at state `s_t`:

```
target_F(s_t)  =  Σ_{k<n} γ^k r_{t+k}
              +  γ^n · ( V_target(s_{t+n}) + Φ(s_{t+n}) )
              -  Φ(s_t)
```

(Or just `G_t − Φ(s_t)` in the pure-MC edge case when the episode ends within
`n` of `t`, since `Φ(terminal) ≡ 0`.) So we apply the shaping **once** in the
dataset by storing the un-shaped per-step rewards alongside Φ(s_t) and
Φ(s_{t+n}); the trainer assembles the shaped target on the fly each batch
(see `blockblaster/train/trainer.py`). The net learns `V_F(s) = V*(s) − Φ(s)`.

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

## Optimizer state is NOT persisted

Checkpoints persist only the model weights and a small bit of metadata —
Adam's optimizer state (`exp_avg`, `exp_avg_sq`, `step`) is deliberately
discarded between rounds. An earlier version did persist it, on the theory
that retaining curvature information would speed up the simulate → train
loop. Empirically the opposite happened: after hundreds of epochs the
second-moment estimates grew large, collapsing the effective per-parameter
learning rate to near-zero, and the net stopped responding to gradients
even when the loss said it should move. Combined with `WEIGHT_DECAY`
applied over many epochs, this kept training pinned in a narrow basin and
prevented the loop from improving past one promotion. Resetting Adam each
round pays a small "warmup" cost per `train()` call but keeps the
optimizer's effective LR healthy for the long haul.

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
    one master seed = SIM_SEED + round_num (varies each round)
    τ = SIM_TEMPERATURE (softmax sampling over top-M beam leaves)
    → episodes collected with the stable policy + exploration diversity.

Eval round                (round % EVAL_INTERVAL == 0)
    PAIRED head-to-head on EVAL_SEEDS (e.g. [42, 43, 44, 45, 46]):
        run challenger arm on those seeds with τ = 0 (deterministic)
        run champion   arm on those seeds with τ = 0 (deterministic)
        both arms see identical piece streams, so piece-luck cancels
    promote iff
        per-seed wins / |EVAL_SEEDS|     ≥ PROMOTION_SEED_WIN_FRACTION  AND
        (chal_median - champ_median) / champ_median  ≥ PROMOTION_MEDIAN_MARGIN
```

We compare on **median**, not mean: a single unusually-long lucky episode
can drag the mean far above typical play, and we don't want one tail event
tipping a promotion decision. Median is the typical-game score.

Why the **paired**, **multi-seed**, **margin** gate (versus a single-seed
"median > bar" rule):

- **Single-seed medians are noisy.** With 250 episodes on one seed, the
  sample median has a CI wide enough that two *identical* policies promote
  ~50% of the time — pure coin flip. Multiple seeds + per-seed comparison
  cuts that noise by √K.
- **Pairing cancels piece-stream luck.** Champion and challenger play the
  *same* random piece streams; if the challenger is genuinely better it
  beats champion on most seeds, regardless of how easy/hard that particular
  seed batch is in absolute terms.
- **Margin rejects ties.** Two policies that are statistically equal will
  trade per-seed wins ~50/50; the margin gate requires a clear edge before
  promoting, so we don't churn through "promote, regret, eventually
  re-promote the same weights" cycles.

Eval rounds always use `τ = 0` (deterministic argmax). Data-collection
rounds use `τ = SIM_TEMPERATURE > 0` so the buffer accumulates trajectory
diversity even when the champion never changes — the cure for the previous
"buffer becomes 3000 copies of one fixed policy" failure mode.

Early-round bootstrap: until BEST exists, the first challenger that survives
the eval round is promoted unconditionally (there's no champion to compare
against), seeding BEST so subsequent paired rounds have a real opponent.

Why this design — without it, three failure modes are easy to hit:

1. **Sim-from-latest only** — training instabilities propagate into the
   data-collection policy and `median` can collapse, even though earlier
   weights were good.
2. **Sim-from-best only** — once BEST is set, the simulator never tries the
   actively trained CHECKPOINT, so BEST is effectively frozen.
3. **Deterministic champion data collection** — the buffer becomes
   identical-policy copies and the trainer fits a fixed value function it
   can never improve past; softmax sampling at `τ > 0` injects the needed
   state diversity.

The eval round + paired gate + softmax sampling break all three: most rounds
use the stable champion (no quality regressions), every `EVAL_INTERVAL`th
round gives the challenger a noise-free head-to-head shot at the title, and
exploration keeps the buffer alive between promotions.
