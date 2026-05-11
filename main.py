"""Entry point: launch the pygame Block Blast demo with the trained agent."""

from __future__ import annotations

import param
from blockblaster.gui.app import run
from blockblaster.model.checkpoint import load_if_exists
from blockblaster.model.value_net import ValueNet


def main() -> None:
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

    run(net)


if __name__ == "__main__":
    main()
