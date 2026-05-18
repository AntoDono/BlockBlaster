# Live Assist GUI

[← back to README](../README.md)

`assist.py` (and equivalently `play.py --mode assist`) opens a pygame
side-by-side viewer: the left panel mirrors a live phone screen, and the
right panel shows a reconstructed Block Blast scene built from what the
assist sees plus the move the trained agent recommends.

```
┌────────────────────────────┬────────────────────────────┐
│      PHONE SCREEN          │     RECONSTRUCTED SCENE    │
│   (live mirror with grid + │    (8×8 board + mini       │
│    queue calibration       │     queue with confidence  │
│    overlays and ghost      │     readout + suggested    │
│    placement preview)      │     piece highlighted)     │
└────────────────────────────┴────────────────────────────┘
```

## Calibration

Two calibration boxes are persisted on disk and drawn on top of the phone
panel:

- **Grid** box defines the 8×8 play area.
- **Queue** box defines the strip containing the 3 upcoming pieces.

Drag on the phone panel in the active mode to (re)set the box; press `Tab`
to switch modes. Boxes are stored per platform
(`assist_config_ios.json` / `assist_config_android.json`) so switching
between devices does not wipe your saved calibration.

## Keys

| Key / Button | Action |
|-----|--------|
| `Tab` / Mode chip           | Toggle calibration mode (GRID / PIECES) |
| drag on phone               | Set the bounding box for the active mode |
| `R` / Clear box chip        | Clear the active mode's box |
| `D` / Dump debug chip       | Dump per-slot debug crops + masks + overlays to `assist_debug/` |
| `A` / Auto-play chip        | Toggle auto-play on/off (Android only; **always off at startup**) |
| `Q` / `ESC` / Quit chip     | Quit |

Each queue slot in the recon panel gets a `p=0.XX` readout next to its
number, colour-coded green ≥ 0.90, yellow 0.70–0.90, red < 0.70 — so a
miscalled piece is visually obvious without opening the debug dump.

## Piece classifier

The queue is decoded by a tiny CNN (~300 K params, < 1 ms / crop on CPU)
classifying each of the 3 slot crops into one of `NUM_CLASSES = 43` labels
(42 pieces + empty).

**Trained entirely on synthetic data — no labelled real screenshots.**
[`blockblaster/piece_cnn/synth.py`](../blockblaster/piece_cnn/synth.py)
renders chamfered cell grids in randomised configurations that mirror what
the assist sees in practice:

| Knob | Range / behaviour |
|------|-------------------|
| Slot aspect (w/h) | `[0.55, 1.05]` — biased tall to match real iOS queue crops (~0.78) |
| Piece-to-slot fill | piece's longest axis fills `[0.30, 0.65]` of slot's short side |
| Cell bevel | smooth top→bottom gradient 85% of the time, line-based 15% |
| Cell border | thin (~`0.04–0.10 × cell`), darker shade of the cell colour, occasionally pure black |
| Drop shadow | always (subtle): alpha `[0.10, 0.28]`, offset `[0.05, 0.18] × cell`, gaussian blur |
| Low-contrast regime | 30% of samples: background HSV within ±12° hue / ±50 value of the piece |
| Per-cell colour jitter | small HSV jitter; 20% chance of fully multi-coloured cells |
| Geometric aug | ±7° rotation, ±0.06 shear, occasional blur |
| Photo-style aug | JPEG round-trip (q ∈ [45, 90]), per-channel BGR cast, edge occlusions |

A coarse 4×4 spatial head (no global-average-pool) preserves enough
positional information that the classifier can still tell e.g. `4x1` from
`5x1` after three stride-2 pools.

If `piece_cnn.pt` is missing or fails to load, `PieceRecognizer` transparently
falls back to a projection / template-matching heuristic
([`piece_recognizer.py`](../blockblaster/assist/piece_recognizer.py)) so the
assist GUI still works.

### Piece CNN module layout

The synth renderer is split into focused sub-modules:

```
blockblaster/piece_cnn/
  config.py   # all rendering constants and augmentation knobs
  color.py    # HSV sampling, per-cell colour jitter, background generation
  draw.py     # draw_cell, draw_piece_shadow, JPEG/cast/occlusion corruptions
  synth.py    # render_piece_sample, generate_batch, pregenerate_dataset
  model.py    # PieceCNN architecture + PieceClassifier inference wrapper
  __init__.py # public re-exports
```

## Commands

```bash
uv run train_piece_cnn.py            # ~100k synth samples, ~12 epochs → piece_cnn.pt

uv run play.py --platform ios                    # assist GUI (iOS, read-only)
uv run play.py --platform android --mode assist  # assist GUI (Android AVD / phone)
uv run play.py --platform android                # auto-play (headless)
uv run play.py --platform android --display      # auto-play with live preview
```

`assist.py` (legacy entry point) is equivalent to
`play.py --platform ios --mode assist`.

Retrain `piece_cnn.pt` whenever you add new piece definitions to
`pieces.py` (output dimension changes) or tune the synth distribution.

For driving the phone (touch injection on a physical Android device) see
[android-autoplay.md](android-autoplay.md).
