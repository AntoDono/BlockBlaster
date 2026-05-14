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
    best_median_ever = float("-inf")
    best_median_round = 0

    for round_num in range(1, args.rounds + 1):
        divider = "=" * 50
        print(f"\n{divider}")
        print(f"  ROUND {round_num} / {args.rounds}")
        print(divider)

        # Every EVAL_INTERVAL rounds, evaluate the challenger (latest CHECKPOINT)
        # against the champion (BEST).  Other rounds collect data with the
        # champion so the sim policy never regresses.
        is_eval_round = (round_num % param.EVAL_INTERVAL == 0)
        round_label = "EVAL (challenger)" if is_eval_round else "champion"

        # ── Simulate ────────────────────────────────────────────────────
        print(f"\n[Round {round_num}] Simulating {param.NUM_SIMULATIONS} episodes — {round_label}...")
        t0 = time.time()
        stats = run_simulations(force_checkpoint=is_eval_round)
        print(f"  Done in {time.time() - t0:.1f}s")

        if stats["max"] > best_score_ever:
            best_score_ever = stats["max"]
            best_score_round = round_num

        # Promotion is only meaningful when the source isn't already BEST.
        # That's the case on (a) eval rounds (forced CHECKPOINT), and
        # (b) early normal rounds before BEST exists (resolver falls back
        # to CHECKPOINT).  When sim used BEST itself, we deliberately do
        # NOT update best_median_ever — that prevents sampling noise from
        # ratcheting the champion's bar upward without a real promotion.
        # We compare on MEDIAN (not mean) so a single lucky max-score
        # episode can't tip the decision.
        sim_source = stats.get("checkpoint_path")
        src_path = Path(sim_source) if sim_source else None
        dst_path = Path(param.BEST_CHECKPOINT_PATH)
        promotion_eligible = (
            src_path is not None and src_path.resolve() != dst_path.resolve()
        )

        if promotion_eligible:
            if stats["median"] > best_median_ever:
                prev_best = best_median_ever
                best_median_ever = stats["median"]
                best_median_round = round_num
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                prev_str = "−∞" if prev_best == float("-inf") else f"{prev_best:.1f}"
                print(
                    f"  Challenger WON (median {stats['median']:.1f} > {prev_str}) "
                    f"→ promoted {src_path.name} to {dst_path.name}"
                )
            elif is_eval_round:
                print(
                    f"  Challenger LOST (median {stats['median']:.1f} ≤ {best_median_ever:.1f}); "
                    f"champion ({dst_path.name}) retained"
                )

        print(
            f"  Best ever: max={best_score_ever} (round {best_score_round})  "
            f"median={best_median_ever:.1f} (round {best_median_round})"
        )

        # ── Train ───────────────────────────────────────────────────────
        print(f"\n[Round {round_num}] Training...")
        t0 = time.time()
        train()
        print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n{'=' * 50}")
    print(f"Loop complete after {args.rounds} round(s).")
    print(f"Latest checkpoint:  {param.CHECKPOINT_PATH}")
    if Path(param.BEST_CHECKPOINT_PATH).exists():
        print(f"Best-median snapshot: {param.BEST_CHECKPOINT_PATH}  (median={best_median_ever:.1f}, round {best_median_round})")
    print(
        f"Best score ever: {best_score_ever} (round {best_score_round})  |  "
        f"Best median score: {best_median_ever:.1f} (round {best_median_round})"
    )
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
