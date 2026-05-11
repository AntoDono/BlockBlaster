"""Entry point: run N Block Blast simulations and save trajectories to disk."""

import param
from blockblaster.sim.runner import run_simulations


def main() -> None:
    param.print_params()
    print(f"\nRunning {param.NUM_SIMULATIONS} simulations -> '{param.SIMULATIONS_DIR}/'")
    run_simulations()
    print("Done.")


if __name__ == "__main__":
    main()
