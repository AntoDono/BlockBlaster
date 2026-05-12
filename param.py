import torch

# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
BOARD_SIZE: int = 8
QUEUE_SIZE: int = 3

# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------
REWARD_PER_CELL: float = 1.0
REWARD_PER_LINE: float = 10.0
MULTI_CLEAR_BONUS: dict[int, float] = {1: 0, 2: 20, 3: 50, 4: 100, 5: 200}

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
GAMMA: float = 0.99

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
NUM_SIMULATIONS: int = 500
MAX_SIMULATIONS: int = 1000       # cap on total episodes kept; oldest are deleted when exceeded
MAX_STEPS_PER_EPISODE: int = 3000 # hard cap per episode to prevent infinite games
SIM_EPSILON: float = 0.2          # exploration rate during simulation
SIM_WORKERS: int = 16              # >1 uses multiprocessing
SIMULATIONS_DIR: str = "simulations"
SIM_SEED: int = 42

# ---------------------------------------------------------------------------
# Reward shaping (potential-based: F = gamma*Phi(s') - Phi(s))
# Phi(s) = POTENTIAL_COEFF    * (sum row_fill^2 + sum col_fill^2)
#         - TRANSITIONS_COEFF * (row transitions + col transitions)
#         + FITTABILITY_COEFF * sum_{p in PIECES} |p| * num_legal_placements(p, board)
# Term 1: rewards near-complete rows/columns (line-clear setup).
# Term 2: penalises fragmented boards (many filled<->empty flips per row/col);
#          a solid block [1,1,1,1,0,0,0,0] has 1 transition, a checkerboard
#          [1,0,1,0,1,0,1,0] has 7.  Subtracted from Phi — lower = better.
# Term 3: rewards boards where all piece types still have legal placements
#          (directly penalises e.g. a board where the 3x3 square has no room).
# Applied at the dataset level (training targets only); env rewards are unchanged
# so reported scores remain the real game score.
# ---------------------------------------------------------------------------
POTENTIAL_COEFF: float = 0.07
TRANSITIONS_COEFF: float = 0.1   # penalty on total row+col transitions (subtracted from Phi)
FITTABILITY_COEFF: float = 0.03  # weight on Σ |p| * num_legal_placements(p, board)

# ---------------------------------------------------------------------------
# Policy lookahead
# ---------------------------------------------------------------------------
LOOKAHEAD_DEPTH: int = 3          # pieces to look ahead (= full queue); set to 1 for 1-step greedy
LOOKAHEAD_MAX_BATCH: int = 4096   # max states per net forward pass (bounds GPU memory)
BEAM_WIDTH: int = 5               # beams kept per depth during lookahead (larger = more exhaustive)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
USE_DIHEDRAL_AUG: bool = True     # 8x augmentation via rotations + reflections

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_EPOCHS: int = 5
BATCH_SIZE: int = 512
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 3e-4
TEST_SPLIT: float = 0.2           # fraction of episodes held out for test
EVAL_INTERVAL_EPOCHS: int = 5
SPLIT_SEED: int = 0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
CNN_CHANNELS: int = 64
HIDDEN_SIZE: int = 512

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH: str = "checkpoints/value_net.pt"
BEST_CHECKPOINT_PATH: str = "checkpoints/best_value_net.pt"  # snapshot of weights that produced the best mean score across simulate->train rounds
LOG_INTERVAL: int = 10


def print_params() -> None:
    """Print all uppercase hyperparameters in sorted order."""
    import sys
    module = sys.modules[__name__]
    params = {k: getattr(module, k) for k in dir(module) if k.isupper()}
    width = max(len(k) for k in params)
    print("=" * 50)
    print("Hyperparameters")
    print("=" * 50)
    for k in sorted(params):
        print(f"  {k:<{width}} = {params[k]}")
    print("=" * 50)
