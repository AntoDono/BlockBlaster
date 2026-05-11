"""Alternating simulate → train loop for iterative policy improvement."""

import argparse
import time

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

    for round_num in range(1, args.rounds + 1):
        divider = "=" * 50
        print(f"\n{divider}")
        print(f"  ROUND {round_num} / {args.rounds}")
        print(divider)

        # ── Simulate ────────────────────────────────────────────────────
        print(f"\n[Round {round_num}] Simulating {param.NUM_SIMULATIONS} episodes...")
        t0 = time.time()
        run_simulations()
        print(f"  Done in {time.time() - t0:.1f}s")

        # ── Train ───────────────────────────────────────────────────────
        print(f"\n[Round {round_num}] Training...")
        t0 = time.time()
        train()
        print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n{'=' * 50}")
    print(f"Loop complete after {args.rounds} round(s).")
    print(f"Checkpoint: {param.CHECKPOINT_PATH}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
