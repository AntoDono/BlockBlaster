# BlockBlaster — Monte Carlo Value Network

A Block Blast game engine paired with a CNN state-value network `v(s)`
trained via Monte Carlo returns with **potential-based reward shaping**
(Ng, Harada, Russell 1999) and **D4 symmetry augmentation**. The agent
plays by greedily picking the action whose afterstate has the highest
predicted value (with the shaping potential added back at decision time
so the optimal policy is unchanged).

A live **assist GUI** mirrors a phone screen, recognises the board and
queue via a tiny piece-classifier CNN trained on synthetic data, and
overlays the agent's recommended move. On Android, an **auto-play** mode
closes the loop with a visual servo that drags the piece into place on
the device.

## Quick start

```bash
uv sync                                          # install
uv run simulate.py                               # collect episodes
uv run train.py                                  # fit v(s)
uv run main.py                                   # watch the agent play
uv run play.py --platform ios --mode assist      # live assist on iPhone
uv run play.py --platform android                # auto-play on Android
```

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/game-rules.md](docs/game-rules.md)             | Board / queue / scoring rules and the 42-piece enumeration. |
| [docs/architecture.md](docs/architecture.md)         | Top-level folder layout and per-subpackage maps. |
| [docs/algorithm.md](docs/algorithm.md)               | State encoding, value network, MC pipeline with beam-search lookahead, potential-based reward shaping, D4 augmentation, champion / challenger checkpointing. |
| [docs/hyperparameters.md](docs/hyperparameters.md)   | Every knob in [`param.py`](param.py) with default and meaning. |
| [docs/training.md](docs/training.md)                 | `simulate` → `train` → repeat loop, generated files, watching the trained agent play. |
| [docs/assist-gui.md](docs/assist-gui.md)             | Side-by-side viewer, calibration, key bindings, piece classifier (synth-data trainer + CNN). |
| [docs/android-autoplay.md](docs/android-autoplay.md) | Emulator / physical-phone setup, closed-loop visual servo, scrcpy v1.20 + adbutils touch tunnel. |
| [docs/visual-servo.md](docs/visual-servo.md)         | Deep dive on the placer: two-mode controller, online plant-gain learning, all tunables, retuning recipe. |

The docs cross-reference each other; the README is just the index.
