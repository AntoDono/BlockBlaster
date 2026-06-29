# Algorithm

[← back to README](../README.md)

How the agent learns and selects moves. Files: [`model/`](../blockblaster/model/), [`agent/policy.py`](../blockblaster/agent/policy.py), [`game/potential.py`](../blockblaster/game/potential.py), [`train/`](../blockblaster/train/).

## State encoding

[`model/encoder.py`](../blockblaster/model/encoder.py) turns `(board, queue)` into a `(1 + QUEUE_SIZE, 8, 8)` float tensor:

- **Channel 0:** board occupancy (0/1).
- **Channels 1…3:** each queue piece rasterised into the top-left of an 8×8 plane (zero-padded; empty slots all-zero).

## Value network

[`model/value_net.py`](../blockblaster/model/value_net.py) — a small CNN, `v(s) → scalar`:

```
Conv3×3(in→C) → ReLU → Conv3×3(C→C) → ReLU → Conv3×3(C→C) → ReLU
Flatten → Linear(C·64 → H) → ReLU → Linear(H → 1)
```

with `C = CNN_CHANNELS (16)`, `H = HIDDEN_SIZE (128)`. No pooling — the 8×8 resolution is preserved through the conv stack.

## Potential-based reward shaping

The net is trained to predict the **shaped** value `V_F(s) = V*(s) − Φ(s)`. Following Ng, Harada & Russell (1999), shaping with a potential `Φ` leaves the optimal policy unchanged provided `Φ` is added back at decision time: `V*(s) = V_F(s) + Φ(s)`.

[`game/potential.py`](../blockblaster/game/potential.py) defines `Φ` as three terms (coefficients in [`param.py`](../param.py)):

```
Φ(s) = POTENTIAL_COEFF   · (Σ row_fill² + Σ col_fill²)      # reward near-complete lines
     − TRANSITIONS_COEFF · (row + col filled↔empty flips)   # penalise fragmented boards
     + FITTABILITY_COEFF · Σ_p |p|·legal_placements(p, s)    # keep room for every piece type
```

The fill term gives a dense gradient toward line-clear setups before the agent has ever cleared one; transitions penalise checkerboards; fittability directly punishes boards where a large piece (e.g. the 3×3) can no longer be placed. `Φ(terminal) := 0`.

## Training target: n-step TD + frozen target net

[`train/dataset.py`](../blockblaster/train/dataset.py) emits, per timestep `t`:

```
target_F(s_t) = Σ_{k<n} γ^k r_{t+k}
              + bootstrap · γ^n · (V_target(s_{t+n}) + Φ(s_{t+n}))
              − Φ(s_t)
```

with `n = TD_N_STEP`. When the episode ends within `n` steps (`bootstrap = 0`) this reduces to the pure Monte Carlo return minus `Φ(s_t)`.

[`train/trainer.py`](../blockblaster/train/trainer.py) fits the live net to this target by MSE. `V_target` is a **frozen deep copy** of the net, refreshed every `TARGET_REFRESH_BATCHES` minibatches. Bootstrapping off a moving target net is what lets training keep improving even when the replay buffer is dominated by a single policy — unlike pure MC, whose target is pinned to `V^π_buffer`. (Adam state is intentionally *not* restored across rounds, to avoid stale second-moment estimates freezing the net.)

**D4 augmentation:** states are stored once in a unique-tensor pool; `__getitem__` lazily applies one of the 8 dihedral group elements to both `s_t` and `s_{t+n}` with the *same* rotation, so the target stays well-defined (`V_F` and `Φ` are approximately D4-invariant). Augmentation is train-split only.

## Action selection: 3-piece beam search

[`agent/policy.py → select_action`](../blockblaster/agent/policy.py) scores each candidate move sequence as the **true discounted return** over the full 3-piece queue:

```
score = r_0 + γ·r_1 + γ²·r_2 + γ³·V*(s_3)
```

where each `r_k` is the real placement reward (cells + line bonuses) and `V*(leaf) = V_F(leaf) + Φ(leaf)`. The search:

1. **Depth 0→1:** place piece A at every legal position, score `r_0 + γ·V*`, keep top `BEAM_WIDTH`.
2. **Depth 1→2:** expand each beam with B, score through `γ²·V*`, keep top `BEAM_WIDTH`.
3. **Depth 2→3:** expand with C, collect all leaf candidates with the full return.

Intermediate states are encoded with the *remaining* unplaced pieces as queue context (matching training). All distinct queue orderings are tried; net forward passes are chunked to `LOOKAHEAD_MAX_BATCH`.

**Dead-end handling:** a leaf where the next planned piece has no legal placement is the search's view of *game over* (`V* = 0`, flagged terminal). At selection time, if any non-terminal continuation exists for any first move, only those compete — so a high-reward "suicide" path (clear now, then can't fit the next piece) can't beat a survivable one. Only when *every* move is forced into a terminal leaf do terminal leaves rank against each other.

**Exploration:** with `temperature = 0` (eval) it picks the argmax first-move. With `temperature > 0` (data-collection rounds) it softmax-samples one of the top-`SIM_EXPLORE_TOP_M` distinct first-moves with weight `exp(score/τ)`, injecting state diversity into the buffer. `epsilon` (or a missing net) falls back to a uniform random legal action.

## Why this shape

Earlier versions scored only `V*(s_3)` and dropped the per-placement rewards, which under-weighted line clears *inside* the lookahead window — fatal on cramped late-game boards where clearing now is survival. Scoring the full discounted return fixes that; the dead-end filter prevents reward-greedy self-destruction.
