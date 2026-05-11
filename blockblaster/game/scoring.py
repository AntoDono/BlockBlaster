"""Reward computation helpers for Block Blast."""

from __future__ import annotations

import param


def placement_reward(num_cells_placed: int) -> float:
    return num_cells_placed * param.REWARD_PER_CELL


def clear_reward(num_lines_cleared: int) -> float:
    """Reward for cleared lines including multi-clear bonus."""
    if num_lines_cleared == 0:
        return 0.0
    base = num_lines_cleared * param.REWARD_PER_LINE
    bonus_table = param.MULTI_CLEAR_BONUS
    # Use largest key <= num_lines_cleared for bonus lookup
    key = min(num_lines_cleared, max(bonus_table.keys()))
    bonus = bonus_table.get(key, 0.0)
    return base + bonus


def step_reward(num_cells: int, num_lines: int) -> float:
    return placement_reward(num_cells) + clear_reward(num_lines)
