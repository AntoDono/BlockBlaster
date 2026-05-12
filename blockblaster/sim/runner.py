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
EpisodeStats = tuple[int, int, bool]  # (total_score, episode_length, truncated)


def _worker_init(epsilon: float, sim_dir: str) -> None:
    """Initializer stored as module-level so it can be pickled for mp."""
    pass


def _run_one(args: tuple[int, int, float, str, int, Optional[str]]) -> EpisodeStats:
    """Top-level function (picklable) that runs one episode and saves it."""
    idx, seed, epsilon, sim_dir, checkpoint_epoch, sim_path_str = args
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
    )
    write_episode(traj, sim_dir, idx)
    return (
        int(traj["total_score"]),
        int(traj["episode_length"]),
        bool(traj["truncated"]),
    )


def run_simulations(
    num_simulations: int | None = None,
    epsilon: float | None = None,
    sim_dir: str | None = None,
    workers: int | None = None,
    base_seed: int | None = None,
    force_checkpoint: bool = False,
) -> dict:
    """Run N episodes, save each as a JSON file, and return aggregate stats.

    When `force_checkpoint=True` the simulator loads CHECKPOINT_PATH (the
    challenger) instead of BEST_CHECKPOINT_PATH (the champion); used by the
    `run_loop` to run an evaluation round so a newly trained policy can earn
    promotion to BEST.

    Returned dict:
        {
            "scores": list[int],
            "lengths": list[int],
            "truncated": int,
            "mean": float, "median": float, "max": int, "min": int,
            "checkpoint_path": str | None,  # the checkpoint sim actually loaded
        }
    """
    n = num_simulations if num_simulations is not None else param.NUM_SIMULATIONS
    eps = epsilon if epsilon is not None else param.SIM_EPSILON
    directory = sim_dir or param.SIMULATIONS_DIR
    num_workers = workers if workers is not None else param.SIM_WORKERS
    seed = base_seed if base_seed is not None else param.SIM_SEED

    rng = random.Random(seed)
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(n)]

    # Resolve which checkpoint sim should use ONCE, here, and thread it
    # through to every worker.  This guarantees parent and workers (and the
    # caller, via stats["checkpoint_path"]) all agree on the source even if
    # files on disk change mid-round.
    sim_path = resolve_sim_checkpoint_path(force_checkpoint=force_checkpoint)
    sim_path_str: Optional[str] = str(sim_path) if sim_path else None

    net = ValueNet()
    meta = load_if_exists(net, sim_path_str) if sim_path_str else None
    checkpoint_epoch = meta.get("epoch", 0) if meta else 0

    role = "challenger (CHECKPOINT)" if force_checkpoint else "champion (BEST)"
    src_name = Path(sim_path_str).name if sim_path_str else "random"
    print(f"  Sim policy: {role}  → {src_name}")

    args_list = [
        (i, seeds[i], eps, directory, checkpoint_epoch, sim_path_str)
        for i in range(n)
    ]

    scores: list[int] = []
    lengths: list[int] = []
    truncated_count = 0

    if num_workers > 1:
        # CUDA forbids fork-based re-initialization, so always use 'spawn'.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            for score, length, truncated in tqdm(
                pool.imap_unordered(_run_one, args_list),
                total=n,
                desc="Simulating",
                unit="ep",
            ):
                scores.append(score)
                lengths.append(length)
                truncated_count += int(truncated)
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

        for i in tqdm(range(n), desc="Simulating", unit="ep"):
            traj = run_episode(
                net=net_single,
                epsilon=eps,
                seed=seeds[i],
                policy_label=policy_label,
                checkpoint_epoch=ckpt_epoch,
            )
            write_episode(traj, directory, i)
            scores.append(int(traj["total_score"]))
            lengths.append(int(traj["episode_length"]))
            truncated_count += int(traj["truncated"])

    _trim_oldest(directory, param.MAX_SIMULATIONS)

    stats = {
        "scores": scores,
        "lengths": lengths,
        "truncated": truncated_count,
        "mean": statistics.fmean(scores) if scores else 0.0,
        "median": statistics.median(scores) if scores else 0.0,
        "max": max(scores) if scores else 0,
        "min": min(scores) if scores else 0,
        "checkpoint_path": sim_path_str,
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
