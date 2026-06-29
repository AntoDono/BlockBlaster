<div align="center">

# **BlockBlaster**
## A value-network agent that plays Block Blast on a real phone

*Train a value network in simulation. Recognise the live game from a mirrored screen. Drive the finger with a closed-loop visual servo. Watch the phone play itself.*

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![pygame](https://img.shields.io/badge/pygame--ce-2.5-9C27B0?style=for-the-badge&logo=python&logoColor=white)](https://pyga.me)
[![scrcpy](https://img.shields.io/badge/scrcpy-1.20-000000?style=for-the-badge&logo=android&logoColor=white)](https://github.com/Genymobile/scrcpy)

![BlockBlaster auto-play demo](assets/Blockblaster_Demo.gif)

</div>

---

## What this is

Three real subsystems wired into one loop:

```mermaid
flowchart LR
    subgraph TRAIN["1 · Train v(s)"]
        direction TB
        T1["Pure-Python<br/>simulator"]
        T2["CNN value net"]
        T3["n-step TD targets<br/>+ frozen target net"]
        T4["Potential-based<br/>reward shaping"]
        T5["D4 symmetry aug"]
        T1 --> T2 --> T3 --> T4 --> T5
    end
    subgraph PERCEIVE["2 · Perceive"]
        direction TB
        P1["Live screen mirror"]
        P2["Interactable detection<br/>(bg-subtraction)"]
        P3["Board scanner (HSV 8×8)"]
        P4["Piece CNN (synthetic-trained)"]
        P1 --> P2 --> P3 --> P4
    end
    subgraph ACT["3 · Act"]
        direction TB
        A1["Advisor → suggestion<br/>(greedy on v(s))"]
        A2["Visual servo (PD)"]
        A3["Frame-diff edge tracking"]
        A4["Release on lock"]
        A1 --> A2 --> A3 --> A4
    end
    TRAIN ==> PERCEIVE ==> ACT
    ACT -. closed loop .-> PERCEIVE
```

- **The value network** is a small CNN trained on **n-step TD targets** bootstrapped from a frozen target net, with **potential-based reward shaping** (Ng, Harada & Russell 1999) and **D4 symmetry augmentation**. Moves are chosen by a **3-piece beam search** that scores the true discounted return. An iterative `simulate → train` loop promotes a challenger to champion only when it wins a **paired multi-seed evaluation**.
- **The perception stack** segments the mirrored screen into interactable blobs (background subtraction), scans the board into an 8×8 occupancy grid (HSV), and classifies the three tray pieces with a tiny CNN **trained entirely on synthetic renders**.
- **The visual servo** closes the loop on the device: it frame-diffs the board against a pre-grab baseline to track the held piece by the **edges of its motion blob**, and PD-steps the finger until the piece's footprint is locked onto the advisor's target cells — then lifts.

See [`docs/`](docs/) for the full write-up of each part.

## Quick start

```bash
uv sync                          # install everything

# ── Train the value agent (simulation only) ──
uv run run_loop.py --rounds 10   # iterative simulate → train with promotion gate
uv run main.py                   # watch the trained agent play in a pygame window

# ── Train the piece classifier ──
uv run train_piece_cnn.py        # synth + real data → piece_cnn.pt

# ── Live on a phone ──
uv run play.py --platform ios                   # read-only assist overlay (mirrored iPhone)
uv run play.py --platform android [--serial S]  # assist + on-device auto-play (ADB)
```

In the GUI, press **`A`** to toggle continuous auto-play (Android only — needs touch injection). iOS is read-only (Apple blocks input injection without a paired Mac/Xcode signature).

## Repository tour

| Path | What lives there |
|------|------------------|
| `blockblaster/game/` | Pure-Python simulator: pieces, board, scoring, potential, env. |
| `blockblaster/model/` | State encoder, value-network CNN, checkpoint I/O. |
| `blockblaster/agent/` | `select_action`: 3-piece beam-search policy over `v(s)`. |
| `blockblaster/sim/` | Episode rollout, JSON I/O, multi-process runner. |
| `blockblaster/train/` | n-step TD dataset (+ D4 aug), trainer (target net), logger. |
| `blockblaster/piece_cnn/` | Synthetic renderer + piece classifier CNN + inference wrapper. |
| `blockblaster/assist/` | Live assist app: `vision/` (detect/scan/recognise/advise), `ui/` (pygame app, events, panels), `render/` (panel drawing). |
| `blockblaster/control/` | Device backends (ADB, screenrecord, scrcpy touch, iOS) + `servo.py` closed-loop placer. |
| `blockblaster/gui/` | Standalone offline pygame demo (`main.py`). |
| `docs/` | Subsystem documentation. |

## Documentation

| Doc | Read it for |
|-----|-------------|
| [`docs/architecture.md`](docs/architecture.md) | Folder layout, data flow, entry points. |
| [`docs/game-rules.md`](docs/game-rules.md) | Board / queue / scoring rules and the 42-piece set. |
| [`docs/algorithm.md`](docs/algorithm.md) | Encoder, value net, potential shaping, n-step TD + target net, beam-search policy. |
| [`docs/training.md`](docs/training.md) | The `simulate → train` loop, champion/challenger promotion, generated files. |
| [`docs/hyperparameters.md`](docs/hyperparameters.md) | Every knob in [`param.py`](param.py). |
| [`docs/perception.md`](docs/perception.md) | Interactable detection, board scanner, piece CNN, advisor. |
| [`docs/assist-gui.md`](docs/assist-gui.md) | Panels, key bindings / buttons, auto-play, board editing, debug overlay. |
| [`docs/visual-servo.md`](docs/visual-servo.md) | The closed-loop PD placer: edge tracking, ROI focus, containment. |
| [`docs/control-stack.md`](docs/control-stack.md) | Device backends and the scrcpy v1.20 touch tunnel. |

## Status & limitations

- **Simulation / training:** stable. `run_loop.py`, `main.py`.
- **Piece classifier:** ~99% on held-out real crops; near-deterministic on bar lengths after the resolution-preserving rewrite.
- **iOS assist:** read-only overlay (no input injection).
- **Android auto-play:** works, device-specific. Servo thresholds are tuned for the current capture resolution; the detection morphology and diff thresholds may need a small retune for very different board palettes.
- **Board detection:** auto-detected each frame; press **`E`** to manually draw the board box if detection drifts, **`R`** to recalibrate.
