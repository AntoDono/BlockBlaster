## Simulation Config Presets

Two named presets for `param.py`, optimized for different goals. Only the params listed here differ between them; everything else stays at its default value in `param.py`.

### `default` — fast, cheap rounds

Original tuning. Each round finishes quickly, but episodes tend to die early (one ε-greedy random move on a packed board ends the game), and the narrow beam prunes good multi-clear setups at depth 1.

```python
NUM_SIMULATIONS      = 500
MAX_SIMULATIONS      = 3000
SIM_EPSILON          = 0.05
BEAM_WIDTH           = 10
LOOKAHEAD_MAX_BATCH  = 4096
POTENTIAL_COEFF      = 0.05
TRANSITIONS_COEFF    = 0.2
FITTABILITY_COEFF    = 0.02
```

Use when: iterating on the training loop, debugging, or doing fast smoke tests.

### `quality` — long games, dense data (current)

Goal: episodes survive deep into the game so the trainer sees real late-game states. Compute per round is significantly higher; that's intentional.

```python
NUM_SIMULATIONS      = 1000
MAX_SIMULATIONS      = 8000
SIM_EPSILON          = 0.0
BEAM_WIDTH           = 128
LOOKAHEAD_MAX_BATCH  = 16384
POTENTIAL_COEFF      = 0.10
TRANSITIONS_COEFF    = 0.4
FITTABILITY_COEFF    = 0.05
```

Use when: collecting training data you actually trust.

### Why each change matters

| Param | default | quality | Rationale |
|---|---|---|---|
| `SIM_EPSILON` | `0.05` | `0.0` | ε-greedy picks a uniform-random legal placement 5% of the time. On a 60–80% full board, "random legal" is almost always an un-clearable spot — game ends in a few steps. ε-greedy is the wrong exploration mechanism for Block Blast; diversity should come from seeds and checkpoint rotation, not self-destructive moves. |
| `BEAM_WIDTH` | `10` | `128` | Branching factor is often 100–300 legal placements per piece. Width 10 keeps ~10³ leaves out of >10⁶ at depth 3, and prunes lines that look bad at depth 1 but enable a triple-clear at depth 3. Wider beam = recovers those lines. |
| `LOOKAHEAD_MAX_BATCH` | `4096` | `16384` | Just so the wider beam doesn't get chunked into many small GPU forwards. Drop back to `8192` if you hit OOM — quality is unaffected, only throughput. |
| `POTENTIAL_COEFF` | `0.05` | `0.10` | Stronger reward for near-complete rows/columns. The default was tuned alongside the narrow beam; with the wider beam you want training targets to credit setup states more explicitly. |
| `TRANSITIONS_COEFF` | `0.2` | `0.4` | Harsher penalty on fragmented (checkerboard-like) boards. Late-game survival correlates strongly with low transitions. |
| `FITTABILITY_COEFF` | `0.02` | `0.05` | More weight on "all piece types still have legal placements." Directly penalises boards where the 3×3 square or other large pieces are dead, which is the most common silent failure mode. |
| `NUM_SIMULATIONS` | `500` | `1000` | More episodes per round = better tails (the rare 5000+ score games are where the trainer actually learns endgame). |
| `MAX_SIMULATIONS` | `3000` | `8000` | Bigger retained replay pool so the trainer sees variety across recent champions, not just the last round's distribution. |

### What was deliberately *not* changed

- `LOOKAHEAD_DEPTH = 3` — already equals `QUEUE_SIZE`. Can't go deeper without sampling unknown future pieces.
- `GAMMA = 0.99` — correct for episode lengths in the thousands of steps.
- `MAX_STEPS_PER_EPISODE = 6000` — episodes die from terminal states, not truncation. Raising the cap does nothing until games actually start hitting it.
- Training params (`NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, …) — orthogonal to simulation quality.

### Knobs to try next (params only)

If `quality` isn't enough and you still want more from the param surface alone:

- `BEAM_WIDTH = 256` — diminishing returns above ~128, but free quality if you have the VRAM.
- `NUM_SIMULATIONS = 2000`, `MAX_SIMULATIONS = 20000` — when the data pool is the bottleneck.
- `EVAL_INTERVAL = 2` — promote challengers more often, so the sim policy improves faster between rounds.

### Knobs that need code changes (out of scope for params)

Listed here so you know where the remaining wins are:

1. Add immediate line-clear rewards into beam scoring (currently the lookahead scores only the value of the post-3-piece board, ignoring rewards earned *during* the 3 placements — biggest single win).
2. Replace ε-greedy with temperature-softmax over the top-N final actions (diversity without suicide moves).
3. Dihedral-average the value at inference (8 symmetries, free variance reduction since the net is already trained with `USE_DIHEDRAL_AUG=True`).
4. MC truncated rollouts at beam leaves, or full MCTS/PUCT with V as the leaf evaluator.
