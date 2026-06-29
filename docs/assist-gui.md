# Assist GUI

[← back to README](../README.md)

```bash
uv run play.py --platform ios                    # read-only overlay (mirrored iPhone)
uv run play.py --platform android [--serial S]   # assist + on-device auto-play
```

The app ([`assist/ui/app.py`](../blockblaster/assist/ui/app.py)) opens a fullscreen pygame window with four panels. It pulls frames from a `control.Device`, reads the latest snapshot from the background [analyzer](perception.md), and renders. Auto-play and the visual servo run on a worker thread so the UI never blocks.

## Panels

```
┌──────────────┬─────────────────────┬──────────────┬────────────┐
│ PHONE SCREEN │ RECONSTRUCTED SCENE │  FRAME DIFF  │ PIECE CNN  │
│ live mirror  │ 8×8 board + queue   │ motion +     │ tray crop  │
│ + detection  │ + suggested ghost   │ servo debug  │ → predicted│
│   overlays   │   placement         │   overlay    │   piece    │
└──────────────┴─────────────────────┴──────────────┴────────────┘
        status bar (device, App fps, ADB fps)
        controls (clickable chips)
```

- **Phone screen** ([`render/phone.py`](../blockblaster/assist/render/phone.py)) — the live frame with detected interactables annotated. In *Edit Board* mode you drag the board box here.
- **Reconstructed scene** ([`render/recon.py`](../blockblaster/assist/render/recon.py)) — the scanned 8×8 board, the detected tray pieces, and the advisor's suggested placement drawn as a ghost; caption shows `suggested: <piece> at row R, col C (slot N)`.
- **Frame diff** ([`render/frame_diff.py`](../blockblaster/assist/render/frame_diff.py)) — per-pixel motion over a dimmed frame, the **gold suggestion outline** mapped onto the live board, and (with Debug on) the **servo tracking overlay**.
- **Piece CNN** ([`render/cnn_debug.py`](../blockblaster/assist/render/cnn_debug.py)) — the slot crop fed to the classifier and its prediction + confidence.

## Controls

Clickable chips (bottom bar) and keyboard shortcuts:

| Key / Chip | Action |
|------------|--------|
| `A` / Autoplay | Toggle **continuous** auto-play (Android only; off at startup). Chip turns green when on, shows ● while a placement runs. |
| `D` / Debug | Toggle the servo-tracking overlay on the Frame Diff panel. |
| `E` / Edit Board | Toggle manual board editing — drag a box on the phone panel to override auto-detection. |
| `R` / Recalibrate | Drop all latched state (held suggestion, board/queue latch, advisor cache, cached board) and re-detect from scratch; also clears the manual override. |
| `S` / Screenshot | Save the window to `screenshots/`. |
| `Q` / `Esc` / Quit | Exit. |

## Auto-play

Press **`A`** to arm continuous auto-play (Android only). While armed, the main loop dispatches one [visual-servo](visual-servo.md) placement of the current suggestion at a time on a worker thread, with an `~0.8 s` cooldown after each so the drop animation settles and a fresh suggestion is computed before the next move. It quietly waits when there's no valid suggestion and resumes when one appears.

This pairs with the analyzer's event gating: the servo's own drag motion raises a frame-diff event, which pauses suggestion recompute and keeps the target latched through the move.

## Edit Board

If board auto-detection drifts, press **`E`**, drag a rectangle over the board on the phone panel, and release. The box (converted to frame pixels via the panel's live scale/offset) becomes a **board override** the analyzer uses for scanning and suggestions, drawn as a persistent gold rectangle. Press **`R`** to revert to auto-detection. Mouse coordinates map correctly even in scaled fullscreen (the window uses pygame's `SCALED` flag).

## Debug overlay

With **`D`** on, the Frame Diff panel shows exactly what the servo tracks (see [visual-servo.md](visual-servo.md) for the full meaning):

- **Gold rectangle + 5 rings** — the target footprint and its 5 reference points (centre + corners).
- **Cyan rectangle + 5 dots** — the measured piece extent and its 5 points, each tied to the target by a line (the 5-point error mapping).
- **Magenta cross** — the commanded finger position.
- **Status banner** — `TRAVELING` / `FOCUSED — FINE APPROACH` / `BOUNDARY HIT — PUSHING BACK` / `LOCKED — RELEASING` / `SEARCHING`, colour-coded.
- Text: match score, `err`, applied `corr`, and `dist to target`. Once focused, a dimmed window shows the local region actually being searched (empty cells only).
