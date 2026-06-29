# Hyperparameters

[← back to README](../README.md)

Every tunable lives in [`param.py`](../param.py) (the RL/sim/training side). Values below are the current defaults. Run `python -c "import param; param.print_params()"` to dump them.

## Game

| Param | Default | Meaning |
|-------|---------|---------|
| `BOARD_SIZE` | 8 | Board edge length. |
| `QUEUE_SIZE` | 3 | Pieces visible / lookahead depth. |

## Rewards

| Param | Default | Meaning |
|-------|---------|---------|
| `REWARD_PER_CELL` | 1.0 | Per placed cell. |
| `REWARD_PER_LINE` | 25.0 | Per row/col cleared. |
| `MULTI_CLEAR_BONUS` | `{1:0,2:50,3:150,4:350,5:700}` | Bonus by simultaneous lines cleared. |

## Monte Carlo / discount

| Param | Default | Meaning |
|-------|---------|---------|
| `GAMMA` | 0.99 | Discount factor. |

## Simulation

| Param | Default | Meaning |
|-------|---------|---------|
| `NUM_SIMULATIONS` | 150 | Episodes per round. |
| `MAX_SIMULATIONS` | 1500 | Replay-buffer cap (oldest episodes trimmed). |
| `MAX_STEPS_PER_EPISODE` | 6000 | Hard per-episode cap. |
| `SIM_EPSILON` | 0.0 | ε-greedy uniform exploration (random-policy fallback). |
| `SIM_TEMPERATURE` | 0.15 | Softmax temperature over top-M beam leaves (data-collection rounds). |
| `SIM_EXPLORE_TOP_M` | 5 | Candidate cap for softmax sampling. |
| `SIM_WORKERS` | 16 | Parallel episode workers (`>1` uses multiprocessing). |
| `SIM_SEED` | 42 | Base master seed. |
| `EVAL_INTERVAL` | 3 | Every Nth round is a paired eval round. |

## Promotion gate

| Param | Default | Meaning |
|-------|---------|---------|
| `EVAL_SEEDS` | `[42,43,44,45,46]` | Shared seeds for paired champion/challenger eval. |
| `PROMOTION_SEED_WIN_FRACTION` | 0.6 | Fraction of seeds the challenger must win. |
| `PROMOTION_MEDIAN_MARGIN` | 0.02 | Required overall-median margin (fractional). |

## Reward shaping (`Φ`)

| Param | Default | Useful range | Term |
|-------|---------|--------------|------|
| `POTENTIAL_COEFF` | 0.05 | 0–0.10 | row/col fill² (line-clear setup). |
| `TRANSITIONS_COEFF` | 0.05 | 0–0.30 | filled↔empty flips penalty. |
| `FITTABILITY_COEFF` | 0.005 | 0–0.015 | Σ |p|·legal-placements (keep room). |

> If any term's range dwarfs the line-clear rewards (~15 for a single clear), it dominates the regression and the agent optimises for it instead of clearing lines.

## Policy lookahead

| Param | Default | Meaning |
|-------|---------|---------|
| `LOOKAHEAD_DEPTH` | 3 | Pieces to look ahead (= full queue). |
| `LOOKAHEAD_MAX_BATCH` | 4096 | Max states per net forward pass. |
| `BEAM_WIDTH` | 7 | Beams kept per depth. |

## Targets & augmentation

| Param | Default | Meaning |
|-------|---------|---------|
| `USE_DIHEDRAL_AUG` | True | 8× D4 symmetry augmentation (train split only). |
| `TD_N_STEP` | 5 | n-step TD horizon (large → pure MC). |
| `TARGET_REFRESH_BATCHES` | 5000 | Minibatches between target-net refreshes. |

## Training

| Param | Default | Meaning |
|-------|---------|---------|
| `NUM_EPOCHS` | 2 | Epochs per training run. |
| `BATCH_SIZE` | 2048 | Minibatch size. |
| `LEARNING_RATE` | 1e-4 | Adam LR (fresh Adam each round). |
| `WEIGHT_DECAY` | 3e-4 | Adam weight decay. |
| `TEST_SPLIT` | 0.1 | Episode-level held-out fraction. |
| `EVAL_INTERVAL_EPOCHS` | 5 | Epochs between test-loss evals/checkpoints. |
| `SPLIT_SEED` | 0 | Train/test split seed. |

## Model

| Param | Default | Meaning |
|-------|---------|---------|
| `CNN_CHANNELS` | 16 | Conv channels in the value net. |
| `HIDDEN_SIZE` | 128 | FC hidden width. |

## I/O

| Param | Default | Meaning |
|-------|---------|---------|
| `DEVICE` | cuda if available else cpu | Torch device. |
| `CHECKPOINT_PATH` | `checkpoints/value_net.pt` | Challenger weights. |
| `BEST_CHECKPOINT_PATH` | `checkpoints/best_value_net.pt` | Champion weights. |

> The **piece classifier** and **visual servo** have their own constants (in [`blockblaster/piece_cnn/config.py`](../blockblaster/piece_cnn/config.py) and [`blockblaster/control/servo.py`](../blockblaster/control/servo.py)) — they are not part of `param.py`. See [perception.md](perception.md) and [visual-servo.md](visual-servo.md).
