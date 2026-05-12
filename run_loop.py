"""Alternating simulate → train loop for iterative policy improvement."""

import argparse
import shutil
import time
from pathlib import Path

import param
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

        # On a new best mean score, snapshot the checkpoint that just produced
        # it BEFORE the next training step overwrites CHECKPOINT_PATH.  Gives
        # us a recovery point when the simulate -> train loop drifts.
        if new_best_mean:
            src = Path(param.CHECKPOINT_PATH)
            dst = Path(param.BEST_CHECKPOINT_PATH)
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"  New best mean → snapshotted to {dst}")

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
