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

    # ── Calibration ─────────────────────────────────────────────────────
    # If a champion already exists, measure it ONCE on the eval seed so
    # the very first challenger has a real bar to clear (not −∞).  This
    # is the same seed challengers will use, so it's a fair comparison.
    # Skipped when no BEST exists — in that case the first promotion-
    # eligible round bootstraps the bar itself.
    if Path(param.BEST_CHECKPOINT_PATH).exists():
        print("\n[Calibration] Measuring existing champion on eval seed...")
        t0 = time.time()
        calib_stats = run_simulations(
            force_checkpoint=False, base_seed=param.SIM_SEED
        )
        best_median_ever = calib_stats["median"]
        best_median_round = 0
        if calib_stats["max"] > best_score_ever:
            best_score_ever = calib_stats["max"]
        print(
            f"  Champion baseline: median={best_median_ever:.1f}  "
            f"max={calib_stats['max']}  (challengers must beat median to promote)"
        )
        print(f"  Done in {time.time() - t0:.1f}s")

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
        # Eval rounds use a FIXED seed (param.SIM_SEED) so champion and
        # challenger are evaluated on identical piece sequences — a fair,
        # noise-free A/B.  Data-collection (champion) rounds vary the seed
        # by round_num, otherwise SIM_EPSILON=0 + deterministic greedy
        # policy produces byte-identical episodes every round and the
        # replay pool stops growing in variety.
        round_seed = param.SIM_SEED if is_eval_round else param.SIM_SEED + round_num
        print(f"\n[Round {round_num}] Simulating {param.NUM_SIMULATIONS} episodes — {round_label}...")
        t0 = time.time()
        stats = run_simulations(force_checkpoint=is_eval_round, base_seed=round_seed)
        print(f"  Done in {time.time() - t0:.1f}s")

        if stats["max"] > best_score_ever:
            best_score_ever = stats["max"]
            best_score_round = round_num

        # Promotion is only meaningful when the source isn't already BEST.
        # That's the case on (a) eval rounds (forced CHECKPOINT, run on
        # SIM_SEED) and (b) early rounds before BEST exists (resolver
        # falls back to CHECKPOINT).  Champion data-collection rounds run
        # on varied seeds (SIM_SEED + round_num) — their medians are NOT
        # comparable to challengers' (different piece sequences), so they
        # never update the bar.  The bar is set by the startup calibration
        # and then advances only on actual promotions, where the new
        # champion's measured-on-SIM_SEED median replaces it.
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
