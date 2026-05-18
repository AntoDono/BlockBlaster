# Project Architecture

[← back to README](../README.md)

## Top-level layout

```
BlockBlaster/
  param.py             # all hyperparameters (single source of truth) — see hyperparameters.md
  simulate.py          # run N episodes → save JSON trajectories
  train.py             # load trajectories, fit v(s), save checkpoint
  train_piece_cnn.py   # synth-data trainer for the queue piece classifier
  main.py              # pygame demo using the trained agent
  assist.py            # live phone-mirror assist GUI (legacy entry point)
  play.py              # unified entry: ios assist / android assist / android auto-play
  simulations/         # generated — one JSON file per episode (git-ignored)
  checkpoints/         # generated — value_net.pt (git-ignored)
  piece_cnn.pt         # generated — queue piece classifier weights (git-ignored)
  blockblaster/
    game/              # rules engine (board, pieces, scoring, potential, env)
    model/             # encoder + value CNN + checkpoint I/O
    agent/             # greedy + beam-search policy
    sim/               # rollout, episode I/O, multi-process runner
    train/             # dataset (MC returns + D4 aug), trainer, logger
    gui/               # standalone pygame app for offline play
    piece_cnn/         # queue piece classifier (synth renderer + CNN + inference)
    assist/            # live-mirror assist GUI (board + queue scanner, advisor, overlays)
    control/           # device I/O: ADB, scrcpy, visual servo, auto-player loop
```

## Per-subpackage maps

### `blockblaster/game/`

```
pieces.py       # 42 piece definitions + sampling
board.py        # Board: place, clear lines, can_place, is_game_over
scoring.py      # reward = cells placed + line bonus + multi-clear bonus
potential.py    # Phi(s) for potential-based reward shaping
env.py          # BlockBlastEnv: reset / step / clone / legal_actions
```

### `blockblaster/model/`

```
encoder.py      # (board, queue) → (4, 8, 8) float tensor
value_net.py    # small CNN → scalar value
checkpoint.py   # save / load (incl. Adam optimizer state)
```

### `blockblaster/agent/`

```
policy.py       # greedy 1-step lookahead + ε-exploration; adds Phi(s')
```

### `blockblaster/sim/`

```
rollout.py      # single episode → trajectory dict
io.py           # write / read episode JSON
runner.py       # run N episodes (optionally multiprocessing, spawn-safe)
```

### `blockblaster/train/`

```
dataset.py      # EpisodeDataset: shaped MC returns + D4 augmentation
trainer.py      # fit v_theta, eval on test split, save best checkpoint
logger.py       # epoch logging helpers
```

### `blockblaster/assist/`

```
app.py             # top-level pygame app (mode dispatch)
app_events.py      # keyboard / mouse / chip event handling
app_overlay.py     # HUD chips, target markers, ghost overlays
app_state.py       # AppState dataclass (calibration, runtime flags)
app_autoplay.py    # auto-play loop wired into the assist GUI
device_stream.py   # iOS frame source via tunneld / DVT
calibration.py     # persisted grid + queue bounding boxes
scanner.py         # crop calibrated 8×8 grid → Board (+ ghost detection)
piece_recognizer.py# queue-slot → Piece via CNN (with heuristic fallback)
piece_mask.py      # cell-occupancy mask extraction
piece_debug.py     # per-slot debug crops dumped to assist_debug/
advisor.py         # wraps the value net for one-shot move suggestions
analyzer.py        # higher-level frame → advice glue
layout.py          # window / panel rects
render.py          # phone panel, recon panel, status bar, overlays
render_phone.py    # phone mirror panel drawing
render_recon.py    # reconstructed scene panel drawing
```

### `blockblaster/control/`

See [android-autoplay.md → Control module layout](android-autoplay.md#control-module-layout)
for the device I/O stack and the scrcpy-based visual servo.

### `blockblaster/piece_cnn/`

See [assist-gui.md → Piece classifier](assist-gui.md#piece-classifier) for the
synth-data trainer and inference wrapper.

## Where to look next

- **How the agent decides moves** → [algorithm.md](algorithm.md)
- **What every tunable does** → [hyperparameters.md](hyperparameters.md)
- **How to train it** → [training.md](training.md)
- **Running the assist GUI** → [assist-gui.md](assist-gui.md)
- **Driving a physical phone** → [android-autoplay.md](android-autoplay.md)
