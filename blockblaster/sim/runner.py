"""Orchestrate running N simulations, optionally in parallel."""

from __future__ import annotations

import multiprocessing as mp
import random
from typing import Optional

from tqdm import tqdm

import param
from blockblaster.model.checkpoint import load_if_exists
from blockblaster.model.value_net import ValueNet
from blockblaster.sim.io import list_episodes, write_episode
from blockblaster.sim.rollout import run_episode


def _worker_init(epsilon: float, sim_dir: str) -> None:
    """Initializer stored as module-level so it can be pickled for mp."""
    pass


def _run_one(args: tuple[int, int, float, str, int]) -> None:
    """Top-level function (picklable) that runs one episode and saves it."""
    idx, seed, epsilon, sim_dir, checkpoint_epoch = args
    net: Optional[ValueNet] = None
    net_obj = ValueNet()
    meta = load_if_exists(net_obj, param.CHECKPOINT_PATH)
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


def run_simulations(
    num_simulations: int | None = None,
    epsilon: float | None = None,
    sim_dir: str | None = None,
    workers: int | None = None,
    base_seed: int | None = None,
) -> None:
    """Run N episodes and save each as a JSON file."""
    n = num_simulations if num_simulations is not None else param.NUM_SIMULATIONS
    eps = epsilon if epsilon is not None else param.SIM_EPSILON
    directory = sim_dir or param.SIMULATIONS_DIR
    num_workers = workers if workers is not None else param.SIM_WORKERS
    seed = base_seed if base_seed is not None else param.SIM_SEED

    rng = random.Random(seed)
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(n)]

    # Load checkpoint metadata to know what epoch we're at
    net = ValueNet()
    meta = load_if_exists(net)
    checkpoint_epoch = meta.get("epoch", 0) if meta else 0

    args_list = [
        (i, seeds[i], eps, directory, checkpoint_epoch)
        for i in range(n)
    ]

    if num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            for _ in tqdm(
                pool.imap_unordered(_run_one, args_list),
                total=n,
                desc="Simulating",
                unit="ep",
            ):
                pass
        _trim_oldest(directory, param.MAX_SIMULATIONS)
    else:
        # Single-process: reuse one loaded net for efficiency
        net_single: Optional[ValueNet] = None
        policy_label = "random"
        ckpt_epoch = 0
        net_obj = ValueNet()
        meta_single = load_if_exists(net_obj)
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

    _trim_oldest(directory, param.MAX_SIMULATIONS)


def _trim_oldest(directory: str, max_episodes: int) -> None:
    """Delete the oldest episode files if total exceeds max_episodes."""
    files = list_episodes(directory)  # already sorted oldest-first
    excess = len(files) - max_episodes
    if excess > 0:
        for f in files[:excess]:
            f.unlink()
        print(f"  Trimmed {excess} old episode(s) (kept {max_episodes})")
