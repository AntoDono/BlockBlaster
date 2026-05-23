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
REWARD_PER_LINE: float = 25.0
MULTI_CLEAR_BONUS: dict[int, float] = {1: 0, 2: 50, 3: 150, 4: 350, 5: 700}

# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
GAMMA: float = 0.99

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
NUM_SIMULATIONS: int = 250
MAX_SIMULATIONS: int = 3000       # cap on total episodes kept; oldest are deleted when exceeded
MAX_STEPS_PER_EPISODE: int = 6000 # hard cap per episode to prevent infinite games
SIM_EPSILON: float = 0.0           # ε-greedy uniform exploration; superseded by SIM_TEMPERATURE for net-driven exploration. Kept for the random-policy fallback when no checkpoint exists.
# Boltzmann sampling over the top-M final beam leaves during DATA-COLLECTION
# rounds.  τ=0.0 reproduces deterministic argmax (eval rounds always pass 0.0).
# τ is in score units (board reward scale); 0.3 is mild — top-M candidates
# typically span 1-20 score points so τ=0.3 gives meaningful but soft sampling.
SIM_TEMPERATURE: float = 0.15
SIM_EXPLORE_TOP_M: int = 5         # cap on # of top final-leaf candidates eligible for softmax sampling; restricts exploration to moves the search already rated highly.
SIM_WORKERS: int = 16              # >1 uses multiprocessing
SIMULATIONS_DIR: str = "simulations"
SIM_SEED: int = 42
EVAL_INTERVAL: int = 3            # every Nth round, sim runs the CHECKPOINT (challenger) instead of BEST (champion); challenger promoted to BEST iff its median beats the current best.

# ── Paired multi-seed evaluation (promotion gate) ───────────────────────────
# Eval rounds run BOTH champion and challenger on the same set of base seeds,
# splitting NUM_SIMULATIONS evenly across them.  Per-seed medians are compared
# pairwise so piece-stream luck cancels.  Promotion requires:
#   - challenger beats champion's per-seed median on ≥ PROMOTION_SEED_WIN_FRACTION of seeds, AND
#   - challenger's overall median exceeds champion's by ≥ PROMOTION_MEDIAN_MARGIN (fractional, e.g. 0.02 = 2%).
EVAL_SEEDS: list[int] = [42, 43, 44, 45, 46]
PROMOTION_SEED_WIN_FRACTION: float = 0.6
PROMOTION_MEDIAN_MARGIN: float = 0.02

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
# Reference magnitudes for comparison (8x8 board):
#   - 1-line clear reward  ≈ 15   (REWARD_PER_LINE + cells_placed)
#   - 2-line clear reward  ≈ 45
#   - 3-line clear reward  ≈ 85
# Each coefficient below is multiplied by an unscaled per-board quantity whose
# min..max range is annotated; the "Phi range" column is what actually shows up
# in the training target.  If any line's Phi range dwarfs the line-clear
# rewards above, that term will dominate the regression and the agent will
# optimise for it instead of for actually clearing lines.

# Raw term:  Σ row_fill² + Σ col_fill²              (0 .. 1024)
# Useful coef range: 0.0 .. 0.10  (0=off, 0.05=mild, 0.10=strong, >0.10 dominates)
POTENTIAL_COEFF: float = 0.05

# Raw term:  total row+col transitions               (0 .. 112)
# Useful coef range: 0.0 .. 0.30  (0=off, 0.10=mild, 0.30=strong, >0.30 dominates)
TRANSITIONS_COEFF: float = 0.05

# Raw term:  Σ_{p ∈ PIECES} |p| · num_legal_placements(p, board)   (0 .. ~6828)
# Useful coef range: 0.0 .. 0.015 (0=off, 0.005=mild, 0.015=strong, >0.02 dominates)
# Current value 0.08 is ~5× past "dominates" — almost certainly why the agent
# optimises for "keep the board empty" instead of for line clears.
FITTABILITY_COEFF: float = 0.005

# ---------------------------------------------------------------------------
# Policy lookahead
# ---------------------------------------------------------------------------
LOOKAHEAD_DEPTH: int = 3          # pieces to look ahead (= full queue); set to 1 for 1-step greedy
LOOKAHEAD_MAX_BATCH: int = 4096  # max states per net forward pass (bounds GPU memory)
BEAM_WIDTH: int = 7              # beams kept per depth during lookahead (larger = more exhaustive)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
USE_DIHEDRAL_AUG: bool = True     # 8x augmentation via rotations + reflections

# ---------------------------------------------------------------------------
# Target (n-step TD + target network)
# ---------------------------------------------------------------------------
# Training target: target(s_t) = Σ_{k<n} γ^k r_{t+k}  +  γ^n V_target(s_{t+n})
# when the episode lasts ≥ n more steps; otherwise a pure MC return.  V_target
# is a frozen copy of the net refreshed every TARGET_REFRESH_BATCHES batches,
# so targets actually shift during training (unlike pure MC, which freezes
# once the net has fit the buffer's policy).  Set TD_N_STEP very large to
# fall back to full-episode MC behaviour while keeping the new pipeline.
TD_N_STEP: int = 5
TARGET_REFRESH_BATCHES: int = 5000

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_EPOCHS: int = 5
BATCH_SIZE: int = 1024
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 3e-4
TEST_SPLIT: float = 0.2           # fraction of episodes held out for test
EVAL_INTERVAL_EPOCHS: int = 5
SPLIT_SEED: int = 0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
CNN_CHANNELS: int = 16
HIDDEN_SIZE: int = 128

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH: str = "checkpoints/value_net.pt"
BEST_CHECKPOINT_PATH: str = "checkpoints/best_value_net.pt"  # CHAMPION: sim loads this on normal rounds; CHECKPOINT_PATH is loaded every EVAL_INTERVAL rounds as challenger and promoted here iff it beats the champion's median. Falls back to CHECKPOINT_PATH then random on early rounds before BEST exists.
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
