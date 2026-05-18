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
| `SIM_EPSILON` | `0.2` | Exploration rate during simulation |
| `SIM_WORKERS` | `8` | `>1` enables multiprocessing (always spawn-mode for CUDA safety) |
| `SIMULATIONS_DIR` | `"simulations"` | Output folder for episode JSONs |
| `SIM_SEED` | `42` | Seed for episode seed generation |
| `EVAL_INTERVAL` | `5` | Every Nth round, sim loads CHECKPOINT (challenger) instead of BEST (champion); challenger is promoted iff its median beats the champion's. |
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
