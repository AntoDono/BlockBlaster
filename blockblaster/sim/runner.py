"""Orchestrate running N simulations, optionally in parallel."""

from __future__ import annotations

import multiprocessing as mp
import random
import statistics
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import param
from blockblaster.model.checkpoint import load_if_exists, resolve_sim_checkpoint_path
from blockblaster.model.value_net import ValueNet
from blockblaster.sim.io import list_episodes, write_episode
from blockblaster.sim.rollout import run_episode


# Per-episode summary returned to the parent process so it can aggregate
# round statistics (mean / max score, etc.) without re-reading every JSON.
# `base_seed` is the master seed that spawned this episode's per-episode seed;
# returned so the parent can group scores by base seed for paired comparison.
EpisodeStats = tuple[int, int, bool, int]  # (total_score, episode_length, truncated, base_seed)


def _worker_init(epsilon: float, sim_dir: str) -> None:
    """Initializer stored as module-level so it can be pickled for mp."""
    pass


def _run_one(args: tuple[int, int, int, float, str, int, Optional[str], float]) -> EpisodeStats:
    """Top-level function (picklable) that runs one episode and saves it."""
    idx, seed, base_seed, epsilon, sim_dir, checkpoint_epoch, sim_path_str, temperature = args
    net: Optional[ValueNet] = None
    net_obj = ValueNet()
    meta = load_if_exists(net_obj, sim_path_str) if sim_path_str else None
    if meta is None:
        net = None
        policy_label = "random"
        ckpt_epoch = 0
    else:
        net = net_obj.to(param.DEVICE)
        net.eval()
        policy_label = "greedy_v_theta"
        ckpt_epoch = meta.get("epoch", 0)

    traj = run_episode(
        net=net,
        epsilon=epsilon,
        seed=seed,
        policy_label=policy_label,
        checkpoint_epoch=ckpt_epoch,
        temperature=temperature,
    )
    write_episode(traj, sim_dir, idx)
    return (
        int(traj["total_score"]),
        int(traj["episode_length"]),
        bool(traj["truncated"]),
        int(base_seed),
    )


def run_simulations(
    num_simulations: int | None = None,
    epsilon: float | None = None,
    sim_dir: str | None = None,
    workers: int | None = None,
    base_seed: int | None = None,
    base_seeds: list[int] | None = None,
    force_checkpoint: bool = False,
    sim_path_override: str | None = None,
    index_offset: int = 0,
    role_label: str | None = None,
    temperature: float | None = None,
) -> dict:
    """Run N episodes, save each as a JSON file, and return aggregate stats.

    Seeding:
      - `base_seeds`: list of master seeds; episodes are distributed evenly
        (`num_simulations // len(base_seeds)` per master).  Per-episode seeds
        are derived deterministically from each master via a seeded RNG so two
        calls with the same `base_seeds` produce identical piece streams
        (essential for paired champion/challenger comparison).
      - `base_seed`: legacy single-master form; equivalent to
        `base_seeds=[base_seed]`.  Ignored if `base_seeds` is provided.

    Checkpoint resolution:
      - `sim_path_override`: explicit path to load (highest priority).
      - else `force_checkpoint=True` → CHECKPOINT (challenger).
      - else → BEST (champion), with fallback to CHECKPOINT then random.

    `index_offset` is added to every episode's filename index so two calls in
    the same round don't risk filename collisions in `simulations/`.

    Returned dict:
        {
            "scores": list[int],
            "lengths": list[int],
            "truncated": int,
            "mean": float, "median": float, "max": int, "min": int,
            "checkpoint_path": str | None,
            "per_seed_scores": dict[int, list[int]],   # keyed by master seed
            "per_seed_median": dict[int, float],
        }
    """
    n_requested = num_simulations if num_simulations is not None else param.NUM_SIMULATIONS
    eps = epsilon if epsilon is not None else param.SIM_EPSILON
    directory = sim_dir or param.SIMULATIONS_DIR
    num_workers = workers if workers is not None else param.SIM_WORKERS
    temp = temperature if temperature is not None else param.SIM_TEMPERATURE

    if base_seeds is None:
        base_seeds = [base_seed if base_seed is not None else param.SIM_SEED]
    if not base_seeds:
        raise ValueError("base_seeds must contain at least one seed")

    # Distribute episodes uniformly across masters.  If n_requested isn't
    # divisible by K, drop the remainder so each master gets an equal share —
    # the per-seed medians must be computed over equal-sized samples for the
    # paired comparison to be balanced.
    n_per = max(1, n_requested // len(base_seeds))
    n = n_per * len(base_seeds)

    # (episode_seed, master_seed) pairs.  Each master's RNG is seeded by the
    # master value itself so the same `base_seeds` always yield the same
    # piece streams across champion and challenger calls.
    seeds_with_origin: list[tuple[int, int]] = []
    for master in base_seeds:
        rng = random.Random(master)
        for _ in range(n_per):
            seeds_with_origin.append((rng.randint(0, 2**31 - 1), master))

    # Resolve which checkpoint sim should use ONCE, here, and thread it
    # through to every worker.  This guarantees parent and workers (and the
    # caller, via stats["checkpoint_path"]) all agree on the source even if
    # files on disk change mid-round.
    if sim_path_override is not None:
        sim_path: Optional[Path] = Path(sim_path_override) if Path(sim_path_override).exists() else None
    else:
        sim_path = resolve_sim_checkpoint_path(force_checkpoint=force_checkpoint)
    sim_path_str: Optional[str] = str(sim_path) if sim_path else None

    net = ValueNet()
    meta = load_if_exists(net, sim_path_str) if sim_path_str else None
    checkpoint_epoch = meta.get("epoch", 0) if meta else 0

    if role_label is not None:
        role = role_label
    elif sim_path_override is not None:
        role = f"override ({Path(sim_path_override).name})"
    else:
        role = "challenger (CHECKPOINT)" if force_checkpoint else "champion (BEST)"
    src_name = Path(sim_path_str).name if sim_path_str else "random"
    print(
        f"  Sim policy: {role}  → {src_name}  "
        f"({len(base_seeds)} master seed(s) × {n_per} ep = {n}, τ={temp:.2f})"
    )

    args_list = [
        (i + index_offset, ep_seed, master, eps, directory, checkpoint_epoch, sim_path_str, temp)
        for i, (ep_seed, master) in enumerate(seeds_with_origin)
    ]

    scores: list[int] = []
    lengths: list[int] = []
    truncated_count = 0
    per_seed_scores: dict[int, list[int]] = {m: [] for m in base_seeds}

    if num_workers > 1:
        # CUDA forbids fork-based re-initialization, so always use 'spawn'.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            for score, length, truncated, master in tqdm(
                pool.imap_unordered(_run_one, args_list),
                total=n,
                desc="Simulating",
                unit="ep",
            ):
                scores.append(score)
                lengths.append(length)
                truncated_count += int(truncated)
                per_seed_scores[master].append(score)
    else:
        # Single-process: reuse one loaded net for efficiency
        net_single: Optional[ValueNet] = None
        policy_label = "random"
        ckpt_epoch = 0
        net_obj = ValueNet()
        meta_single = load_if_exists(net_obj, sim_path_str) if sim_path_str else None
        if meta_single is not None:
            net_single = net_obj.to(param.DEVICE)
            net_single.eval()
            policy_label = "greedy_v_theta"
            ckpt_epoch = meta_single.get("epoch", 0)

        for i, (ep_seed, master) in enumerate(tqdm(seeds_with_origin, desc="Simulating", unit="ep")):
            traj = run_episode(
                net=net_single,
                epsilon=eps,
                seed=ep_seed,
                policy_label=policy_label,
                checkpoint_epoch=ckpt_epoch,
                temperature=temp,
            )
            write_episode(traj, directory, i + index_offset)
            scores.append(int(traj["total_score"]))
            lengths.append(int(traj["episode_length"]))
            truncated_count += int(traj["truncated"])
            per_seed_scores[master].append(int(traj["total_score"]))

    _trim_oldest(directory, param.MAX_SIMULATIONS)

    per_seed_median = {
        m: (statistics.median(s) if s else 0.0) for m, s in per_seed_scores.items()
    }

    stats = {
        "scores": scores,
        "lengths": lengths,
        "truncated": truncated_count,
        "mean": statistics.fmean(scores) if scores else 0.0,
        "median": statistics.median(scores) if scores else 0.0,
        "max": max(scores) if scores else 0,
        "min": min(scores) if scores else 0,
        "checkpoint_path": sim_path_str,
        "per_seed_scores": per_seed_scores,
        "per_seed_median": per_seed_median,
    }
    print(
        f"  Score: mean={stats['mean']:.1f}  median={stats['median']:.1f}  "
        f"max={stats['max']}  min={stats['min']}  "
        f"(truncated {truncated_count}/{n})"
    )
    return stats


def _trim_oldest(directory: str, max_episodes: int) -> None:
    """Delete the oldest episode files if total exceeds max_episodes."""
    files = list_episodes(directory)  # already sorted oldest-first
    excess = len(files) - max_episodes
    if excess > 0:
        for f in files[:excess]:
            f.unlink()
        print(f"  Trimmed {excess} old episode(s) (kept {max_episodes})")
