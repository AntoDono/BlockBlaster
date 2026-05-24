"""Entry point: launch the pygame Block Blast demo with the trained agent."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

import param
from blockblaster.gui.app import run
from blockblaster.model.checkpoint import load_if_exists
from blockblaster.model.value_net import ValueNet


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Block Blast agent demo.")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the env (piece stream) and Python/NumPy/Torch RNGs. "
             "Default: 0.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    net = ValueNet().to(param.DEVICE)
    meta = load_if_exists(net)

    if meta is None:
        print(
            f"No checkpoint found at '{param.CHECKPOINT_PATH}'.\n"
            "Running with random policy.  Train first:\n"
            "  uv run simulate.py\n"
            "  uv run train.py\n"
        )
        net = None
    else:
        epoch = meta.get("epoch", "?")
        best  = meta.get("best_test_loss", float("nan"))
        print(
            f"Loaded checkpoint: epoch={epoch}, best_test_loss={best:.4f}\n"
            f"Device: {param.DEVICE}"
        )
        net.eval()

    run(net, seed=args.seed)


if __name__ == "__main__":
    main()
