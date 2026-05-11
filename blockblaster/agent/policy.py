"""Greedy 1-step lookahead policy with epsilon-exploration."""

from __future__ import annotations

import random
from typing import Optional

import torch

import param
from blockblaster.game.env import BlockBlastEnv
from blockblaster.model.encoder import encode_state
from blockblaster.model.value_net import ValueNet


def select_action(
    env: BlockBlastEnv,
    net: Optional[ValueNet],
    epsilon: float = 0.0,
    device: str | None = None,
) -> tuple[int, int, int]:
    """
    Choose a (slot, row, col) action.

    Strategy:
      - Always enumerates legal actions; returns (0,0,0) shouldn't be reached
        since caller checks is_over() first.
      - With probability `epsilon` or if `net` is None: uniform random.
      - Otherwise: simulate each afterstate, batch-evaluate v(s'), pick argmax.
    """
    actions = env.legal_actions()
    if not actions:
        raise RuntimeError("select_action called on a terminal state (no legal actions)")

    if net is None or random.random() < epsilon:
        return random.choice(actions)

    dev = device or param.DEVICE
    net_device = next(net.parameters()).device

    # Build afterstate tensors for all legal actions
    tensors: list[torch.Tensor] = []
    for slot, r, c in actions:
        clone = env.clone()
        clone.step(slot, r, c)
        tensors.append(encode_state(clone.board, clone.queue))

    batch = torch.stack(tensors, dim=0).to(net_device)
    values = net.predict(batch)  # (N,)

    best_idx = int(values.argmax().item())
    return actions[best_idx]
