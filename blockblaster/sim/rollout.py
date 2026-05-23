"""Run a single Block Blast episode and return the trajectory dict."""

from __future__ import annotations

from typing import Optional

import param
from blockblaster.agent.policy import select_action
from blockblaster.game.env import BlockBlastEnv
from blockblaster.model.value_net import ValueNet


def run_episode(
    net: Optional[ValueNet],
    epsilon: float,
    seed: int,
    policy_label: str = "greedy_v_theta",
    checkpoint_epoch: int = 0,
    max_steps: int | None = None,
    device: str | None = None,
    temperature: float = 0.0,
) -> dict:
    """
    Play one episode to completion (or until max_steps is reached).

    Returns a trajectory dict compatible with `blockblaster.sim.io`.
    """
    step_limit = max_steps if max_steps is not None else param.MAX_STEPS_PER_EPISODE
    env = BlockBlastEnv(seed=seed)
    steps: list[dict] = []

    while not env.is_over() and env.steps < step_limit:
        board_snapshot = env.board.to_list()
        queue_ids = [p.piece_id for p in env.queue]

        action = select_action(
            env, net, epsilon=epsilon, device=device, temperature=temperature
        )
        slot, row, col = action
        result = env.step(slot, row, col)

        steps.append({
            "board": board_snapshot,
            "queue": queue_ids,
            "action": [slot, row, col],
            "reward": result.reward,
            "lines_cleared": result.lines_cleared,
        })

    return {
        "seed": seed,
        "policy": policy_label if net is not None else "random",
        "checkpoint_epoch": checkpoint_epoch,
        "total_score": env.total_score,
        "episode_length": env.steps,
        "truncated": env.steps >= step_limit and not env.is_over(),
        "steps": steps,
    }
