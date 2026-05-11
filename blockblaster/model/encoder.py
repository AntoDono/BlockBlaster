"""Encode a board + queue state into a (1 + QUEUE_SIZE, 8, 8) float tensor."""

from __future__ import annotations

import numpy as np
import torch

import param
from blockblaster.game.board import Board
from blockblaster.game.pieces import Piece


def encode_state(board: Board, queue: list[Piece]) -> torch.Tensor:
    """
    Returns a float32 tensor of shape (1 + QUEUE_SIZE, BOARD_SIZE, BOARD_SIZE).

    Channel 0: board occupancy (0.0 / 1.0).
    Channel k (1 <= k <= QUEUE_SIZE): piece k-1 rasterised into the top-left
        corner of an 8x8 plane, zero-padded.  Empty/missing queue slots are
        all-zeros.
    """
    size = param.BOARD_SIZE
    channels = 1 + param.QUEUE_SIZE
    tensor = np.zeros((channels, size, size), dtype=np.float32)

    # Channel 0: board
    tensor[0] = board.grid.astype(np.float32)

    # Channels 1..QUEUE_SIZE: pieces
    for i in range(param.QUEUE_SIZE):
        if i < len(queue):
            piece = queue[i]
            for dr, dc in piece.cells:
                if dr < size and dc < size:
                    tensor[i + 1, dr, dc] = 1.0

    return torch.from_numpy(tensor)


def encode_state_batch(
    boards: list[Board],
    queues: list[list[Piece]],
) -> torch.Tensor:
    """Batch-encode a list of (board, queue) pairs into (N, C, H, W)."""
    tensors = [encode_state(b, q) for b, q in zip(boards, queues)]
    return torch.stack(tensors, dim=0)
