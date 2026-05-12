"""Alternating simulate → train loop for iterative policy improvement."""

import argparse
import shutil
import time
from pathlib import Path

import param
from blockblaster.model.checkpoint import resolve_sim_checkpoint_path
from blockblaster.sim.runner import run_simulations
from blockblaster.train.trainer import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run N rounds of simulate → train."
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=5,
        help="Number of simulate → train iterations (default: 5)",
    )
    args = parser.parse_args()

    param.print_params()
    print(f"\nStarting loop: {args.rounds} round(s)\n")

    best_score_ever = 0
    best_score_round = 0
    best_mean_ever = float("-inf")
    best_mean_round = 0

    for round_num in range(1, args.rounds + 1):
        divider = "=" * 50
        print(f"\n{divider}")
        print(f"  ROUND {round_num} / {args.rounds}")
        print(divider)

        # ── Simulate ────────────────────────────────────────────────────
        print(f"\n[Round {round_num}] Simulating {param.NUM_SIMULATIONS} episodes...")
        t0 = time.time()
        stats = run_simulations()
        print(f"  Done in {time.time() - t0:.1f}s")

        if stats["max"] > best_score_ever:
            best_score_ever = stats["max"]
            best_score_round = round_num
        new_best_mean = stats["mean"] > best_mean_ever
        if new_best_mean:
            best_mean_ever = stats["mean"]
            best_mean_round = round_num
        print(
            f"  Best ever: max={best_score_ever} (round {best_score_round})  "
            f"mean={best_mean_ever:.1f} (round {best_mean_round})"
        )

        # On a new best mean score, promote the source sim used this round
        # to BEST_CHECKPOINT_PATH so future sim rounds stay at this policy.
        # Guard skips the copy when sim already loaded from BEST (src == dst).
        if new_best_mean:
            src_path = resolve_sim_checkpoint_path()
            if src_path is not None:
                dst = Path(param.BEST_CHECKPOINT_PATH)
                if src_path.resolve() != dst.resolve():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst)
                    print(f"  New best mean → snapshotted {src_path.name} to {dst}")

        # ── Train ───────────────────────────────────────────────────────
        print(f"\n[Round {round_num}] Training...")
        t0 = time.time()
        train()
        print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n{'=' * 50}")
    print(f"Loop complete after {args.rounds} round(s).")
    print(f"Latest checkpoint:  {param.CHECKPOINT_PATH}")
    if Path(param.BEST_CHECKPOINT_PATH).exists():
        print(f"Best-mean snapshot: {param.BEST_CHECKPOINT_PATH}  (mean={best_mean_ever:.1f}, round {best_mean_round})")
    print(
        f"Best score ever: {best_score_ever} (round {best_score_round})  |  "
        f"Best mean score: {best_mean_ever:.1f} (round {best_mean_round})"
    )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
