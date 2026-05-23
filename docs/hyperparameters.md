# Hyperparameters

[← back to README](../README.md)

All hyperparameters live in [`param.py`](../param.py) as a single source of
truth. The algorithm-side meaning of each knob is covered in
[algorithm.md](algorithm.md).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BOARD_SIZE` | `8` | Grid dimension |
| `QUEUE_SIZE` | `3` | Pieces shown at once |
| `REWARD_PER_CELL` | `1.0` | Reward for each cell placed |
| `REWARD_PER_LINE` | `10.0` | Reward per line cleared |
| `MULTI_CLEAR_BONUS` | `{1:0, 2:20, 3:50, 4:100, 5:200}` | Extra bonus per clear count |
| `GAMMA` | `0.99` | Discount factor for MC returns |
| `POTENTIAL_COEFF` | `0.07` | Scale of row/col fill term in `Phi` (`0` disables that term) |
| `TRANSITIONS_COEFF` | `0.1` | Penalty weight on row+col transitions in `Phi` (subtracted; `0` disables) |
| `FITTABILITY_COEFF` | `0.03` | Scale of piece-fittability term in `Phi`: `Σ \|p\| × num_placements(p)` (`0` disables) |
| `LOOKAHEAD_DEPTH` | `3` | Pieces to look ahead per move (3 = full queue; 1 = original 1-step greedy) |
| `LOOKAHEAD_MAX_BATCH` | `4096` | Max states per network forward pass during lookahead |
| `BEAM_WIDTH` | `5` | Beams kept per depth during lookahead (larger = more exhaustive, slower) |
| `USE_DIHEDRAL_AUG` | `True` | Apply 8-way D4 augmentation to the training set |
| `NUM_SIMULATIONS` | `500` | Episodes per simulation round |
| `MAX_SIMULATIONS` | `3000` | Cap on stored episodes; oldest are deleted past this |
| `SIM_EPSILON` | `0.0` | ε-greedy uniform exploration. Superseded by `SIM_TEMPERATURE` for net-driven exploration; kept for the random-policy fallback when no checkpoint exists. |
| `SIM_TEMPERATURE` | `0.15` | Boltzmann temperature (in score units) for sampling among the top-M final beam leaves during **data-collection** rounds. `0.0` reproduces deterministic argmax. Eval rounds always use `0.0` regardless of this setting so paired champion/challenger comparisons stay noise-free. |
| `SIM_EXPLORE_TOP_M` | `5` | Cap on number of top-scoring final-leaf candidates eligible for softmax sampling. Restricts exploration to moves the search already rated highly, so we get state diversity without committing obviously bad late-game moves. |
| `SIM_WORKERS` | `8` | `>1` enables multiprocessing (always spawn-mode for CUDA safety) |
| `SIMULATIONS_DIR` | `"simulations"` | Output folder for episode JSONs |
| `SIM_SEED` | `42` | Seed for episode seed generation |
| `EVAL_INTERVAL` | `5` | Every Nth round, sim loads CHECKPOINT (challenger) instead of BEST (champion); challenger is promoted iff it passes the paired gate (see `EVAL_SEEDS`, `PROMOTION_SEED_WIN_FRACTION`, `PROMOTION_MEDIAN_MARGIN`). |
| `EVAL_SEEDS` | `[42, 43, 44, 45, 46]` | Master seeds used for paired champion-vs-challenger evaluation on promotion rounds. `NUM_SIMULATIONS` is split evenly across them (e.g. 5 seeds × 50 eps = 250). Both arms play identical per-episode seeds derived from each master, so piece-stream luck cancels in the paired comparison. |
| `PROMOTION_SEED_WIN_FRACTION` | `0.6` | Challenger must beat champion's per-seed median on at least this fraction of `EVAL_SEEDS` to be promoted. `>0.5` requires a clear majority and rejects ties. |
| `PROMOTION_MEDIAN_MARGIN` | `0.02` | Additional gate: challenger's overall median across all eval episodes must exceed champion's by at least this *fractional* margin (e.g. `0.02` = 2%). Prevents promoting equal policies on coin-flip wins. |
| `NUM_EPOCHS` | `50` | Training epochs per `train.py` run |
| `BATCH_SIZE` | `256` | Minibatch size |
| `LEARNING_RATE` | `1e-3` | Adam learning rate |
| `WEIGHT_DECAY` | `1e-4` | Adam weight decay |
| `TEST_SPLIT` | `0.1` | Fraction of episodes held out for test |
| `TD_N_STEP` | `5` | n-step horizon for the TD bootstrap. Target = sum of `n` real discounted rewards plus γ^n times `V_target(s_{t+n})`. `1` = pure 1-step TD (low variance, high bias); set ≥ `MAX_STEPS_PER_EPISODE` to fall back to full-episode MC behaviour. |
| `TARGET_REFRESH_BATCHES` | `5000` | How often (in training minibatches) to copy live weights → `V_target`. Lower = faster propagation but less stable; higher = more stable but slower to track the live net's improvements. ~5000 ≈ roughly one refresh per epoch on the current buffer. |
| `EVAL_INTERVAL_EPOCHS` | `5` | Evaluate on test set every N epochs |
| `SPLIT_SEED` | `0` | Seed for train/test episode split |
| `CNN_CHANNELS` | `32` | Convolutional channel width `C` |
| `HIDDEN_SIZE` | `256` | FC hidden layer width `H` |
| `DEVICE` | auto | `"cuda"` if available, else `"cpu"` |
| `CHECKPOINT_PATH` | `"checkpoints/value_net.pt"` | Where to save the model |
| `LOG_INTERVAL` | `10` | Print training stats every N epochs |
