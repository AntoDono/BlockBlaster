# BlockBlaster — Monte Carlo Value Network

A Block Blast game engine paired with a CNN state-value network `v(s)` trained
via Monte Carlo returns.  The agent plays by greedily picking the action whose
afterstate has the highest predicted value.

---

## Game Rules

| Rule | Detail |
|------|--------|
| Grid | 8 × 8, starts empty |
| Queue | 3 pieces shown at once; a fresh batch of 3 is drawn when all are placed |
| Placement | A piece may be placed anywhere its cells fit on empty squares |
| Line clear | Any fully-filled row **and** column clears simultaneously after each placement |
| Multi-clear bonus | Extra reward when ≥ 2 lines clear from a single placement |
| Game over | When no queued piece can be legally placed anywhere |

Pieces: 32 canonical shapes — single cell, bars (1×2 … 1×5, 2×1 … 5×1), 2×2 and 3×3
squares, L/J/T/S/Z variants, and a plus shape.

---

## Architecture

```
BlockBlaster/
  param.py          # all hyperparameters (single source of truth)
  simulate.py       # run N episodes → save JSON trajectories
  train.py          # load trajectories, fit v(s), save checkpoint
  main.py           # pygame demo using the trained agent
  simulations/      # generated — one JSON file per episode (git-ignored)
  checkpoints/      # generated — value_net.pt (git-ignored)
  blockblaster/
    game/
      pieces.py     # 32 piece definitions + sampling
      board.py      # Board: place, clear lines, can_place, is_game_over
      scoring.py    # reward = cells placed + line bonus + multi-clear bonus
      env.py        # BlockBlastEnv: reset / step / clone / legal_actions
    model/
      encoder.py    # (board, queue) → (4, 8, 8) float tensor
      value_net.py  # small CNN → scalar value
      checkpoint.py # save / load / load_if_exists
    agent/
      policy.py     # greedy 1-step lookahead + ε-exploration
    sim/
      rollout.py    # single episode → trajectory dict
      io.py         # write / read episode JSON
      runner.py     # run N episodes (optionally multiprocessing)
    train/
      dataset.py    # EpisodeDataset: load JSONs, compute returns, split
      trainer.py    # fit v_theta, eval on test split, save best checkpoint
      logger.py     # epoch logging helpers
    gui/
      render.py     # pygame draw functions (board, queue panel, info bar)
      app.py        # main game loop (auto-play, pause, reset)
```

---

## Algorithm

### State encoding

`encoder.encode_state(board, queue)` returns a `(4, 8, 8)` float32 tensor:

| Channel | Content |
|---------|---------|
| 0 | Board occupancy (0 = empty, 1 = filled) |
| 1 | Queued piece #1 rasterised into top-left of 8×8 plane |
| 2 | Queued piece #2 rasterised into top-left of 8×8 plane |
| 3 | Queued piece #3 rasterised into top-left of 8×8 plane |

### Value network

```
Conv2d(4 → C, 3×3, pad=1) → ReLU
Conv2d(C → C, 3×3, pad=1) → ReLU
Conv2d(C → C, 3×3, pad=1) → ReLU
Flatten → Linear(C×64 → H) → ReLU → Linear(H → 1)
```

`C = CNN_CHANNELS`, `H = HIDDEN_SIZE` (see `param.py`).

### Monte Carlo training pipeline

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

The action space is small (≤ 3 × 64 = 192 candidates per step), so the policy
is implemented as an **explicit afterstate enumeration**: clone the env for
every legal action, evaluate `v(s')` in a single batched forward pass, and
pick the argmax.

MC returns `G_t = Σ_{k≥t} γ^(k−t) · r_k` align the value function directly
with "survive as long as possible", which is the stated objective.

Iterating simulate → train improves data quality over rounds:

```
Round 1: simulate (random) → train → checkpoint v1
Round 2: simulate (greedy v1 + ε) → train → checkpoint v2
Round 3: simulate (greedy v2 + ε) → train → checkpoint v3
  ...
```

---

## Hyperparameters (`param.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BOARD_SIZE` | `8` | Grid dimension |
| `QUEUE_SIZE` | `3` | Pieces shown at once |
| `REWARD_PER_CELL` | `1.0` | Reward for each cell placed |
| `REWARD_PER_LINE` | `10.0` | Reward per line cleared |
| `MULTI_CLEAR_BONUS` | `{1:0, 2:20, 3:50, 4:100, 5:200}` | Extra bonus per clear count |
| `GAMMA` | `0.99` | Discount factor for MC returns |
| `NUM_SIMULATIONS` | `200` | Episodes per `simulate.py` run |
| `SIM_EPSILON` | `0.2` | Exploration rate during simulation |
| `SIM_WORKERS` | `1` | `>1` enables multiprocessing |
| `SIMULATIONS_DIR` | `"simulations"` | Output folder for episode JSONs |
| `SIM_SEED` | `42` | Seed for episode seed generation |
| `NUM_EPOCHS` | `50` | Training epochs per `train.py` run |
| `BATCH_SIZE` | `256` | Minibatch size |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `WEIGHT_DECAY` | `1e-4` | Adam weight decay |
| `TEST_SPLIT` | `0.1` | Fraction of episodes held out for test |
| `EVAL_INTERVAL_EPOCHS` | `5` | Evaluate on test set every N epochs |
| `SPLIT_SEED` | `0` | Seed for train/test episode split |
| `CNN_CHANNELS` | `32` | Convolutional channel width `C` |
| `HIDDEN_SIZE` | `256` | FC hidden layer width `H` |
| `DEVICE` | auto | `"cuda"` if available, else `"cpu"` |
| `CHECKPOINT_PATH` | `"checkpoints/value_net.pt"` | Where to save the model |
| `LOG_INTERVAL` | `10` | Print training stats every N epochs |

---

## Usage

### Install

```bash
uv sync
```

### Step 1 — Generate simulations

```bash
uv run simulate.py
```

Creates `simulations/ep_*.json` (200 episodes by default).
If no checkpoint exists yet, a **random** policy is used.

### Step 2 — Train

```bash
uv run train.py
```

Loads all episodes in `simulations/`, computes MC returns, trains `v(s)`,
and saves the best checkpoint to `checkpoints/value_net.pt`.
Prints hyperparameters and per-epoch train/test loss.

### Step 3 — Repeat (optional, improves quality)

```bash
uv run simulate.py   # now uses trained checkpoint + ε-exploration
uv run train.py      # fit again on the expanded dataset
# ... repeat as desired ...
```

### Step 4 — Watch the agent play

```bash
uv run main.py
```

Opens a pygame window.  The agent auto-plays using the trained value network.

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `R` | Reset the game |
| `Q` / `ESC` | Quit |

### Generated files

| Path | Description |
|------|-------------|
| `simulations/ep_*.json` | Episode trajectories (git-ignored) |
| `checkpoints/value_net.pt` | Best model checkpoint (git-ignored) |
