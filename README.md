<div align="center">

# **BlockBlaster**
## An end-to-end Monte Carlo value agent that plays Block Blast on a real phone

*Train a value network in simulation. Recognise the live game from a mirrored screen. Drive the finger via a closed-loop visual servo. Watch the phone play itself.*

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![pygame](https://img.shields.io/badge/pygame--ce-2.5-9C27B0?style=for-the-badge&logo=python&logoColor=white)](https://pyga.me)
[![ADB](https://img.shields.io/badge/ADB-Android-3DDC84?style=for-the-badge&logo=android&logoColor=white)](https://developer.android.com/tools/adb)
[![scrcpy](https://img.shields.io/badge/scrcpy-1.20-000000?style=for-the-badge&logo=android&logoColor=white)](https://github.com/Genymobile/scrcpy)

### Demo

<video src="assets/Blockblaster%20Demo.mp4" controls width="720"></video>

> If the embed doesn't play, grab it directly: [`assets/Blockblaster Demo.mp4`](assets/Blockblaster%20Demo.mp4). The agent picks moves, the visual servo lands them on a real Android device.

</div>

---

## What this actually is

Most "Block Blast AI" projects stop at search heuristics or a paper plot of MC returns. This repo runs the whole loop end-to-end:

```
┌────────────────────────┐    ┌────────────────────────┐    ┌────────────────────────┐
│  1. Train v(s)         │    │  2. Perceive game      │    │  3. Act on the device  │
│                        │    │                        │    │                        │
│  • simulator           │    │  • scrcpy mirror       │    │  • advisor → suggestion│
│  • CNN value head      │    │  • board / queue scan  │    │  • visual servo drags  │
│  • Monte Carlo returns │ →  │  • piece classifier    │ →  │  • plant-gain learner  │
│  • potential shaping   │    │  • live assist GUI     │    │  • per-device JSON     │
│  • D4 augmentation     │    │  • ghost recon panel   │    │  • blind-commit guard  │
└────────────────────────┘    └────────────────────────┘    └────────────────────────┘
```

Each stage is a real, runnable subsystem — not a notebook stub. The agent that decides moves is the same agent that plays in simulation; the perception pipeline that draws overlays is the same one that feeds the action loop; the servo that drags pieces learns its own physics constants and persists them per phone.

### Why the pieces are interesting

**The value network** is a small CNN trained via Monte Carlo returns with **potential-based reward shaping** (Ng, Harada & Russell 1999) and **D4 symmetry augmentation** of board states. The agent acts greedily on afterstate value with the shaping potential added back at decision time so the optimal policy is unchanged. See [`docs/algorithm.md`](docs/algorithm.md).

**The perception stack** is built on a tiny piece-classifier CNN trained entirely on synthetic data — every queue tile, every render variant, generated on the fly. The assist GUI overlays the agent's planned move on the mirrored screen in real time and includes a **reconstructed-scene panel** that lets you see what the scanner sees (placed cells, ghost preview, queue confidences). See [`docs/assist-gui.md`](docs/assist-gui.md).

**The visual servo** is the bit most projects skip. It closes the loop on the device: watches the held piece as it's being dragged, runs a two-mode P controller (fixed-stride coarse, plant-inverted fine), **learns the per-device plant gain online** via EMA with column-snap rejection, persists it to disk per ADB serial, and has a blind-commit fallback for top-row placements where the held piece is rendered outside the scanner's view. The Y-axis bias accounts for per-piece geometry — a vertical 4×1 bar aims the finger to a different row than a horizontal 1×4 even when both target the same board cells. See [`docs/visual-servo.md`](docs/visual-servo.md).

## Quick start

```bash
uv sync                                          # install everything
uv run simulate.py                               # collect MC episodes
uv run train.py                                  # fit v(s) on the dataset
uv run main.py                                   # watch the trained agent play in-sim
uv run play.py --platform ios --mode assist      # live overlay on a mirrored iPhone
uv run play.py --platform android                # full auto-play on Android
```

The Android path is the headline feature: connect a phone over ADB, launch Block Blast, run the command, and the agent will calibrate the board, pick moves, drive the finger, and learn its own servo constants over the first few placements. Per-device gains are cached at `learned_device_params/<serial>.json` and reloaded automatically on the next run.

## Repository tour

| Path | What lives there |
|------|------------------|
| `blockblaster/game/` | Pure-Python Block Blast simulator: pieces, board, scoring, legal-move generation. |
| `blockblaster/model/` | State encoder and value-network architecture. |
| `blockblaster/agent/` | Decision policy: afterstate enumeration, value lookup, beam-search lookahead. |
| `blockblaster/piece_cnn/` | Synthetic data generator and CNN that classifies queue tiles from pixels. |
| `blockblaster/assist/` | Pygame assist GUI, screen analyzer, board/queue scanner, recon panel, advisor wiring. |
| `blockblaster/control/` | Device abstractions (ADB, scrcpy), visual servo package, calibration. |
| `blockblaster/control/visual_servo/` | `placer.py` orchestration · `controller.py` two-mode P · `plant_gain.py` online learner + per-device cache · `detection.py` piece-on-board recognition · `tunables.py` every constant. |
| `docs/` | Long-form documentation per subsystem (see below). |
| `learned_device_params/` | Auto-generated JSON cache of per-device servo plant gains. |

## Documentation

The README is just the index. Each doc cross-references the others, so any one of them is a reasonable entry point depending on what you came for.

| Doc | Read it for |
|-----|-------------|
| [`docs/game-rules.md`](docs/game-rules.md) | Board / queue / scoring rules and the 42-piece enumeration. |
| [`docs/architecture.md`](docs/architecture.md) | Top-level folder layout and per-subpackage maps. |
| [`docs/algorithm.md`](docs/algorithm.md) | State encoding, value network, Monte Carlo pipeline with beam-search lookahead, potential-based reward shaping, D4 augmentation, champion / challenger checkpointing. |
| [`docs/hyperparameters.md`](docs/hyperparameters.md) | Every knob in [`param.py`](param.py) with its default and meaning. |
| [`docs/training.md`](docs/training.md) | `simulate` → `train` → repeat loop, generated files, how to watch the trained agent play. |
| [`docs/assist-gui.md`](docs/assist-gui.md) | Side-by-side viewer, calibration flow, key bindings, the synthetic-data piece classifier. |
| [`docs/android-autoplay.md`](docs/android-autoplay.md) | Emulator / physical-phone setup, scrcpy v1.20 + adbutils touch tunnel, end-to-end auto-play. |
| [`docs/visual-servo.md`](docs/visual-servo.md) | Deep dive on the placer: two-mode controller, online plant-gain learning, blind-commit fallback, per-piece Y bias, all tunables, retuning recipe. |

## Status & limitations

Honest about what works and what doesn't:

- **Simulation pipeline:** stable. Train, evaluate, watch in-sim.
- **iOS assist (read-only overlay):** works on a mirrored iPhone — pure visualisation, no input injection (Apple doesn't allow it without a paired Mac/Xcode signature).
- **Android auto-play:** working but device-specific. The servo learns plant gain online; first few placements on a new device may need a couple of retries before the cache converges. Top-row placements of vertical pieces previously failed; now handled by per-piece geometry-aware finger targeting.
- **Calibration:** semi-manual on first use — drop the grid + queue boxes on the mirrored frame once, persisted to JSON for subsequent runs.

If you fork this and play with a different game, the perception + control split is reusable: the servo doesn't know anything about Block Blast specifically, only about "drag this finger so that thing on screen lines up with that target."
