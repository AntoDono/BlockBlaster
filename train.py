"""Entry point: train the ValueNet on stored simulation trajectories."""

import param
from blockblaster.train.trainer import train


def main() -> None:
    param.print_params()
    print()
    train()


if __name__ == "__main__":
    main()
