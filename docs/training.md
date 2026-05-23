# Training the Value Network

[← back to README](../README.md)

For the algorithm side (state encoding, n-step TD targets, target network,
reward shaping, paired champion/challenger evaluation) see
[algorithm.md](algorithm.md). For knobs see
[hyperparameters.md](hyperparameters.md).

## Install

```bash
uv sync
```

## Step 1 — Generate simulations

```bash
uv run simulate.py
```

Creates `simulations/ep_*.json` (`NUM_SIMULATIONS` episodes per round). If no
checkpoint exists yet, a **random** policy is used.

## Step 2 — Train

```bash
uv run train.py
```

Loads all episodes in `simulations/`, builds per-step n-step TD samples
(`s_t`, `s_{t+n}`, partial reward sums), trains `v(s)` against targets
bootstrapped off a periodically-refreshed target network, and saves the best
checkpoint to `checkpoints/value_net.pt`. Prints hyperparameters and
per-epoch train/test loss. See
[algorithm.md → n-step TD training pipeline](algorithm.md#n-step-td-training-pipeline)
for the target math and `TD_N_STEP` / `TARGET_REFRESH_BATCHES` in
[hyperparameters.md](hyperparameters.md) for the relevant knobs.

## Step 3 — Repeat (optional, improves quality)

```bash
uv run simulate.py   # now uses trained checkpoint + ε-exploration
uv run train.py      # fit again on the expanded dataset
# ... repeat as desired ...
```

This is the simulate → train loop described in
[algorithm.md → n-step TD training pipeline](algorithm.md#n-step-td-training-pipeline);
paired multi-seed champion/challenger evaluation is described in
[algorithm.md → Champion / challenger checkpointing](algorithm.md#champion--challenger-checkpointing).

## Step 4 — Watch the agent play

```bash
uv run main.py
```

Opens a pygame window. The agent auto-plays using the trained value network.

| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `R` | Reset the game |
| `Q` / `ESC` | Quit |

## Generated files

| Path | Description |
|------|-------------|
| `simulations/ep_*.json` | Episode trajectories (git-ignored) |
| `checkpoints/value_net.pt` | **Challenger** — latest training checkpoint, updated every train round; loaded by sim on eval rounds (git-ignored) |
| `checkpoints/best_value_net.pt` | **Champion** — stable simulation policy; loaded by sim on normal rounds; only updated when the challenger beats it in an eval round (git-ignored) |
| `piece_cnn.pt` | Queue piece classifier weights, trained on synthetic data (git-ignored) — see [assist-gui.md → Piece classifier](assist-gui.md#piece-classifier) |
