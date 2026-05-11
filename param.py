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
MAX_STEPS_PER_EPISODE: int = 2000 # hard cap per episode to prevent infinite games
SIM_EPSILON: float = 0.2          # exploration rate during simulation
SIM_WORKERS: int = 8              # >1 uses multiprocessing
SIMULATIONS_DIR: str = "simulations"
SIM_SEED: int = 42

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_EPOCHS: int = 10
BATCH_SIZE: int = 256
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 3e-4
TEST_SPLIT: float = 0.2           # fraction of episodes held out for test
EVAL_INTERVAL_EPOCHS: int = 5
SPLIT_SEED: int = 0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
CNN_CHANNELS: int = 32
HIDDEN_SIZE: int = 256

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH: str = "checkpoints/value_net.pt"
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
