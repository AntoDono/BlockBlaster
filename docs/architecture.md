# Architecture

[← back to README](../README.md)

## Top-level layout

```
BlockBlaster/
  param.py             # all RL/sim/training hyperparameters (single source of truth)
  simulate.py          # run NUM_SIMULATIONS episodes → JSON trajectories
  train.py             # fit the value net on stored trajectories
  run_loop.py          # alternating simulate → train loop with promotion gate
  main.py              # offline pygame demo with the trained agent
  play.py              # live assist GUI launcher (ios / android)
  train_piece_cnn.py   # synthetic + real data trainer for the piece classifier
  eval_recognizers.py  # compare geometric vs CNN piece recognition on data/pieces/
  eval_heldout.py      # honest held-out CNN accuracy on the real-crop split

  simulations/         # generated — one JSON per episode (trimmed to MAX_SIMULATIONS)
  checkpoints/         # generated — value_net.pt (challenger) + best_value_net.pt (champion)
  piece_cnn.pt         # generated — piece classifier weights
  data/pieces/<label>/ # collected real piece crops (one folder per piece)

  blockblaster/
    game/    model/    agent/    sim/    train/    piece_cnn/    gui/
    assist/  control/
```

## Two halves

The project splits cleanly into an **offline RL half** and a **live phone half**, joined only by the trained value-net checkpoint.

### Offline RL (trains `v(s)`)

```
game/                      model/                 agent/
  pieces.py  42 pieces       encoder.py  state→tensor   policy.py  beam search
  board.py   place/clear     value_net.py  CNN→scalar
  scoring.py reward           checkpoint.py save/load
  potential.py Φ(s)
  env.py     reset/step/legal_actions

sim/                       train/
  rollout.py one episode     dataset.py  n-step TD samples + D4 aug
  io.py      episode JSON     trainer.py  fit to TD targets (frozen target net)
  runner.py  N eps (mp)       logger.py
```

Flow: `run_loop.py` → `sim/runner.run_simulations` (each worker calls `agent/policy.select_action` per step, writes a trajectory) → `train/trainer.train` (loads trajectories via `train/dataset`, fits `model/value_net`). See [algorithm.md](algorithm.md) and [training.md](training.md).

### Live phone (plays the real game)

```
assist/
  vision/                          ui/                    render/
    detection.py interactables       app.py   main loop      phone.py    mirror panel
    scanner.py   board → 8×8         events.py keys/mouse    recon.py    reconstructed scene
    piece_recognizer.py → Piece      state.py  AppState      cnn_debug.py
    piece_mask.py geometric mask     overlay.py controls     frame_diff.py motion + servo debug
    analyzer.py  background worker    layout.py panel rects
    calibration.py (legacy box I/O)
  advisor.py     value net → Suggestion
  collector.py   real-crop capture tool

control/
  device.py            Device protocol + make_device()
  android_screenrecord.py / android_adb.py   capture + tap/swipe backends
  scrcpy_control.py    persistent scrcpy v1.20 touch session (servo gestures)
  ios_readonly.py      iOS mirror (read-only)
  servo.py             closed-loop PD placement
  coords.py            piece/cell → pixel anchor helpers
```

Flow: `play.py` → `assist/ui/app.run`. The UI loop pulls frames from a `control.Device`, a background `assist/vision/analyzer.AnalysisWorker` detects the board + tray pieces and asks `assist/advisor.Advisor` for a `Suggestion`, and the panels render it. Pressing **A** runs `control/servo.place` on a worker thread to execute the suggestion on-device. See [perception.md](perception.md), [assist-gui.md](assist-gui.md), [visual-servo.md](visual-servo.md), [control-stack.md](control-stack.md).

## The one shared artifact

`checkpoints/best_value_net.pt` (the champion value net) is consumed by:

- `assist/advisor.Advisor`, which loads it (as `model.pt` by default) and exposes `suggest(board_grid, queue) → Suggestion`.

Everything else on the live side is perception/control and knows nothing about the RL internals — the servo only knows "drag the finger so this blob lines up with that target."

## Entry points

| Command | Does |
|---------|------|
| `uv run simulate.py` | One batch of self-play episodes → `simulations/`. |
| `uv run train.py` | One training run on stored episodes → `checkpoints/`. |
| `uv run run_loop.py -r N` | N rounds of simulate→train with the promotion gate. |
| `uv run main.py [--seed S]` | Offline pygame demo with the champion net. |
| `uv run play.py --platform {ios,android}` | Live assist GUI (+ Android auto-play). |
| `uv run train_piece_cnn.py` | Train `piece_cnn.pt`. |
