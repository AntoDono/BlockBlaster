"""Alternating simulate → train loop for iterative policy improvement."""

import argparse
import shutil
import time
from pathlib import Path

import param
from blockblaster.sim.runner import run_simulations
from blockblaster.train.trainer import train


def _paired_eval_round(round_num: int) -> dict:
    """Run a paired champion-vs-challenger evaluation on a shared seed set
    and promote the challenger if it wins per the configured gate.

    Both arms play the same per-episode seeds (derived deterministically from
    `param.EVAL_SEEDS`) so piece-stream luck cancels in the per-seed median
    comparison.  Returns the challenger's stats dict (used by the caller to
    track best-ever max score).
    """
    eval_seeds = list(param.EVAL_SEEDS)
    if not eval_seeds:
        raise ValueError("param.EVAL_SEEDS must contain at least one seed")

    print(
        f"\n[Round {round_num}] Paired eval over {len(eval_seeds)} master seed(s): "
        f"{eval_seeds}"
    )

    # --- Challenger arm ----------------------------------------------------
    t0 = time.time()
    chal_stats = run_simulations(
        force_checkpoint=True,
        base_seeds=eval_seeds,
        index_offset=0,
        role_label="challenger (CHECKPOINT)",
        temperature=0.0,
    )
    chal_path = chal_stats.get("checkpoint_path")
    print(f"  Challenger arm done in {time.time() - t0:.1f}s")

    # If the challenger checkpoint doesn't exist yet (very early rounds), skip.
    if chal_path is None:
        print("  No challenger checkpoint yet; skipping promotion gate.")
        return chal_stats

    dst_path = Path(param.BEST_CHECKPOINT_PATH)

    # --- Bootstrap: no champion → promote unconditionally ------------------
    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chal_path, dst_path)
        print(
            f"  Bootstrap promotion: no champion yet → "
            f"copied {Path(chal_path).name} → {dst_path.name}"
        )
        return chal_stats

    # If challenger source IS already BEST, promotion is a no-op.
    if Path(chal_path).resolve() == dst_path.resolve():
        print(
            "  Challenger and champion paths are identical; skipping gate."
        )
        return chal_stats

    # --- Champion arm (same seeds) -----------------------------------------
    t0 = time.time()
    champ_stats = run_simulations(
        sim_path_override=str(dst_path),
        base_seeds=eval_seeds,
        index_offset=param.NUM_SIMULATIONS,  # avoid filename collisions with challenger arm
        role_label="champion (BEST)",
        temperature=0.0,
    )
    print(f"  Champion arm done in {time.time() - t0:.1f}s")

    # --- Paired comparison -------------------------------------------------
    chal_med = chal_stats["per_seed_median"]
    champ_med = champ_stats["per_seed_median"]

    seed_wins = 0
    print("  Per-seed medians (challenger vs champion):")
    for s in eval_seeds:
        c = chal_med.get(s, 0.0)
        b = champ_med.get(s, 0.0)
        mark = "W" if c > b else ("T" if c == b else "L")
        if c > b:
            seed_wins += 1
        print(f"    seed {s:>10}: {c:>9.1f}  vs  {b:>9.1f}   [{mark}]")

    n_seeds = len(eval_seeds)
    win_frac = seed_wins / n_seeds
    champ_overall = champ_stats["median"]
    chal_overall = chal_stats["median"]
    margin = (chal_overall - champ_overall) / max(champ_overall, 1e-9)

    seeds_ok = win_frac >= param.PROMOTION_SEED_WIN_FRACTION
    margin_ok = margin >= param.PROMOTION_MEDIAN_MARGIN
    promote = seeds_ok and margin_ok

    print(
        f"  Gate: per-seed wins {seed_wins}/{n_seeds} = {win_frac:.2f} "
        f"(need ≥ {param.PROMOTION_SEED_WIN_FRACTION:.2f})  |  "
        f"overall median margin {margin*100:+.2f}% "
        f"(need ≥ {param.PROMOTION_MEDIAN_MARGIN*100:.2f}%)"
    )

    if promote:
        shutil.copy2(chal_path, dst_path)
        print(
            f"  Challenger PROMOTED → copied {Path(chal_path).name} → {dst_path.name}"
        )
    else:
        reasons = []
        if not seeds_ok:
            reasons.append("per-seed wins")
        if not margin_ok:
            reasons.append("median margin")
        print(
            f"  Challenger LOST on: {', '.join(reasons)}; "
            f"champion ({dst_path.name}) retained"
        )

    return chal_stats


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

    for round_num in range(1, args.rounds + 1):
        divider = "=" * 50
        print(f"\n{divider}")
        print(f"  ROUND {round_num} / {args.rounds}")
        print(divider)

        is_eval_round = (round_num % param.EVAL_INTERVAL == 0)

        # ── Simulate ────────────────────────────────────────────────────
        # Eval rounds: paired multi-seed champion vs challenger; promotion
        # decided per `_paired_eval_round`.
        # Data-collection rounds: champion plays with a per-round-varied
        # master seed (single seed → all NUM_SIMULATIONS episodes share that
        # master via deterministic per-episode RNG) so new state diversity
        # enters the replay buffer.
        if is_eval_round:
            stats = _paired_eval_round(round_num)
        else:
            print(f"\n[Round {round_num}] Simulating {param.NUM_SIMULATIONS} episodes — champion (data collection)...")
            t0 = time.time()
            stats = run_simulations(
                force_checkpoint=False,
                base_seeds=[param.SIM_SEED + round_num],
            )
            print(f"  Done in {time.time() - t0:.1f}s")

        if stats["max"] > best_score_ever:
            best_score_ever = stats["max"]
            best_score_round = round_num

        print(f"  Best ever: max={best_score_ever} (round {best_score_round})")

        # ── Train ───────────────────────────────────────────────────────
        print(f"\n[Round {round_num}] Training...")
        t0 = time.time()
        train()
        print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n{'=' * 50}")
    print(f"Loop complete after {args.rounds} round(s).")
    print(f"Latest checkpoint:  {param.CHECKPOINT_PATH}")
    if Path(param.BEST_CHECKPOINT_PATH).exists():
        print(f"Champion snapshot:  {param.BEST_CHECKPOINT_PATH}")
    print(f"Best score ever: {best_score_ever} (round {best_score_round})")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
