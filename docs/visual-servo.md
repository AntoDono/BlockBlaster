# Visual servo — closed-loop placement on Android

[← back to README](../README.md) · [← back to android-autoplay.md](android-autoplay.md)

This document walks through how auto-play physically places a piece on
the device: how the servo loop is structured, what it observes, how the
two-mode controller sizes corrections, how the online plant-gain
estimator works, and where every tunable lives.

If you just want to *use* it, [android-autoplay.md](android-autoplay.md)
already covers setup. This page is for understanding or modifying the
servo itself.

- [Why a servo](#why-a-servo)
- [Algorithm at a glance](#algorithm-at-a-glance)
- [Detection: same signal as the recon panel](#detection-same-signal-as-the-recon-panel)
- [The two-mode controller](#the-two-mode-controller)
- [Plant-gain learning](#plant-gain-learning)
- [Coarse open-loop jump](#coarse-open-loop-jump)
- [Anti-windup, locks, aborts](#anti-windup-locks-aborts)
- [Module layout](#module-layout)
- [Tunables reference](#tunables-reference)
- [Debug output](#debug-output)
- [Retuning for a new device](#retuning-for-a-new-device)

## Why a servo

Earlier versions used a pre-baked **per-axis calibration** — fit
`finger_px → device_px` from a handful of taps, then apply that mapping
forever. That works briefly and breaks the moment anything drifts (the
phone rotates, a Block Blast update changes board layout, scrcpy crops
the capture differently, etc.). Worse, Block Blast does **not** render
the held piece at the finger position: it floats above the finger by a
distance-dependent offset, so even a perfect finger calibration leaves
a residual that grows with drag distance.

A closed-loop servo skips all of that. We never *assume* a mapping; we
observe where the piece actually is and correct directly. The only
thing that needs to be calibrated is the board / queue bounding boxes
(in [assist-gui.md](assist-gui.md)).

## Algorithm at a glance

```
DOWN on queue slot                               # holding the piece
HOLD for HOLD_MS                                 # so Block Blast registers grab, not flick
MOVE finger up by INITIAL_LIFT_PX                # so piece visibly lifts above finger
MOVE finger by (undershoot × distance-to-target) # coarse open-loop jump
loop while time-remaining:
    frame ← wait for fresh device frame
    piece_cells ← scan_board(frame) − initial_placed
    if piece_cells == expected_cells:           # exact lock
        lift, return success
    if piece_anchor ≈ target_anchor:            # tolerant lock
        lift, return success
    update plant-gain estimate from (Δfinger, Δpiece)
    raw_err ← target_anchor − piece_anchor
    finger += axis_step(raw_err, plant_gain)    # two-mode controller
    MOVE finger
abort: drag back over queue and lift            # piece returns
```

The whole loop is ~80 lines of Python in
[`placer.py`](../blockblaster/control/visual_servo/placer.py). Every
piece of logic outside the loop body lives in a sibling module so this
file stays focused on flow.

## Detection: same signal as the recon panel

The held piece **is just a board cell** from the camera's point of view —
Block Blast draws the dragged piece on top of the board, so
[`scan_board`](../blockblaster/assist/scanner.py) reads its cells as
filled. To isolate the moving piece from the permanently-placed blocks,
the servo snapshots `placed_cells_before_grab` once at the start, then
on every iteration:

```
piece_cells = scan_board(frame) − initial_placed
```

This is the entire detection layer. No HSV ghost band, no "valid landing"
preview hack, no separate CNN. The reconstructed scene in the GUI is
rendered from the same `scan_board` output, so:

> If the GUI's recon panel says the piece is on the target cell, the
> servo agrees by construction.

That property is the reason the placement feels reliable — you can see
exactly what the controller sees.

Detection lives in
[`detection.py`](../blockblaster/control/visual_servo/detection.py).

## The two-mode controller

Block Blast's plant gain (piece-px per finger-px) is **>1** and grows
with drag distance. A single fixed P-gain therefore either oscillates
(too high) or stalls on big errors (too low). The controller splits the
problem at `FINE_THRESHOLD_PX = 30`:

| Regime | When | Step size |
|---|---|---|
| Coarse | \|raw_err\| > 30 px | fixed `±STEP_CLAMP_PX` toward target |
| Fine   | \|raw_err\| ≤ 30 px | `FINE_SAFETY_FACTOR × raw / plant_gain` |

In the coarse regime the controller is **plant-blind** on purpose — we
always make `STEP_CLAMP_PX` of finger progress per iteration, so a bad
early plant estimate can't stall convergence. The piece moves
`plant × STEP_CLAMP_PX` px per iter, which traverses the whole board
inside the `MAX_LOOP_S = 2.5 s` budget even at minimal plant gain.

In the fine regime the controller **inverts the plant**: to move the
piece `raw` px, command the finger `raw / plant` px. With a correct
plant estimate this is one-shot convergence; with a slightly low
estimate the `FINE_SAFETY_FACTOR = 0.85` ensures we approach the target
from one side rather than ring around it.

This is in
[`controller.py`](../blockblaster/control/visual_servo/controller.py).

## Plant-gain learning

The plant gain is approximately constant for a given device / game
version but unknown ahead of time, and small finger nudges still produce
useful evidence. The servo therefore **measures** it online:

```python
# every iteration, with prev_* held from the previous iteration:
df = finger_now - finger_prev   # how far we commanded the finger
dp = piece_now  - piece_prev    # how far the piece actually moved
if |df| ≥ 4 and df × dp > 0:    # noise + sticky-frame filter
    sample = clamp(dp / df, 0.4, 4.0)
    plant_g = 0.6 × plant_g + 0.4 × sample      # EMA
```

The filters matter: `|df| < 4 px` is detection noise, and `df × dp ≤ 0`
means the piece moved the *wrong way* (sticky frame, board edge, mis-
detection) — we don't learn from those.

### Persistence across placements

At the end of every `place_with_servo` call (success **or** failure,
inside `finally`), the per-axis estimates are written to module-scope
variables `_learned_plant_gx` / `_learned_plant_gy`. The next call
seeds its in-loop estimate from the cache, and the coarse jump
(see [next section](#coarse-open-loop-jump)) uses it directly.

Effect across a session:

| Placement # | Coarse undershoot used | Why |
|---|---|---|
| 1 | `COARSE_FALLBACK = 0.55` | no learning yet |
| 2 | `0.92 / ~1.8 ≈ 0.51` | learned from #1's many iterations |
| 3+ | `~0.50` | EMA has converged |

The cache is **process-local** — quit the GUI and it resets. If you
want to flush it mid-session (e.g. switching devices), call
`blockblaster.control.visual_servo.plant_gain.reset_learned()` from
a REPL or trap your own keybinding.

Learning subsystem lives in
[`plant_gain.py`](../blockblaster/control/visual_servo/plant_gain.py).

## Coarse open-loop jump

Why we don't teleport the finger straight to the target anchor: the
piece moves `plant × finger_displacement`, so a `100 %` finger jump
moves the piece `plant × 100 %` of the distance — typically 180–250 %.
The piece flies off the board, often loses the grab.

The ideal open-loop jump is `1 / plant` of the distance (piece lands
exactly on target), de-rated by `COARSE_SAFETY = 0.92` so we always
undershoot slightly. The closed loop closes an undershoot easily; an
overshoot requires reversing direction, which is harder to learn from
because the piece's motion gets sluggish near boundaries.

Both the fraction and the in-loop estimator are seeded from the same
learned cache, so the system bootstraps from one `COARSE_FALLBACK = 0.55`
guess on placement #1 and converges within 2–3 placements.

## Anti-windup, locks, aborts

* **Y-axis anti-windup.** Block Blast renders the held piece *above*
  the finger with a growing offset, but only up to a point — past the
  bottom of the board, downward commands stop translating into piece
  motion and just risk losing the grab. We cap finger Y at
  `target_y + FINGER_OVERTRAVEL_Y = target_y + 350 px`.

* **Lock criteria.** Either of these counts as a lock; once we see two
  consecutive frames satisfying one, we settle for `PRE_LIFT_MS = 120`
  and lift:
  1. **Exact**: `piece_cells == expected_cells`.
  2. **Tolerant**: same cell count and piece anchor within
     `LOCK_TOLERANCE_PX = 12` of target on both axes (covers ±1-cell
     flicker on a visually-correct drop).

* **Sub-pixel lock.** If the controller computes step `(0, 0)` while the
  raw error is ≤ 2 px on both axes for two consecutive iterations, we
  accept that as locked too — the gain rounded the correction to zero
  because we're already there.

* **Abort path.** On any of `budget exceeded`, `piece never appeared`,
  `piece shape mismatch`, or `no frames from device`, the servo drags
  the finger back over the queue slot and lifts. A frame outside the
  board has no held-piece cells and Block Blast typically returns the
  piece to the queue, so callers can safely retry on the next analysis.

## Module layout

```
blockblaster/control/visual_servo/
├── __init__.py    # re-exports place_with_servo, ServoResult, _GRAB_Y_NUDGE_PX
├── tunables.py    # every magic constant + ServoResult dataclass
├── plant_gain.py  # online estimator + persistent learned cache
├── controller.py  # axis_step + Y anti-windup
├── detection.py   # piece-cells extraction + lock predicate
└── placer.py      # orchestration (place_with_servo)
```

Public surface (what callers import):

```python
from blockblaster.control.visual_servo import (
    place_with_servo,   # the placement entrypoint
    ServoResult,        # what it returns
    _GRAB_Y_NUDGE_PX,   # used by app_autoplay to draw the swipe arrow
)
```

Internal contract: nothing outside the package should import from the
submodules directly — those are implementation details and may move
around. The `__init__` re-exports are the stable surface.

## Tunables reference

All constants live in
[`tunables.py`](../blockblaster/control/visual_servo/tunables.py) with
per-constant comments explaining the failure mode each one pins. Quick
index by category:

| Category | Constants |
|---|---|
| Gesture bookends | `HOLD_MS`, `PRE_LIFT_MS`, `GRAB_Y_NUDGE_PX`, `INITIAL_LIFT_PX` |
| Loop pacing | `MAX_LOOP_S`, `FRAME_TIMEOUT_S`, `POST_MOVE_SETTLE_MS`, `MAX_NO_PIECE_FRAMES` |
| Lock criteria | `STABLE_MATCHES`, `LOCK_TOLERANCE_PX` |
| Controller | `STEP_CLAMP_PX`, `FINE_THRESHOLD_PX`, `FINE_SAFETY_FACTOR` |
| Anti-windup | `FINGER_OVERTRAVEL_Y` |
| Plant-gain estimator | `PLANT_GAIN_INIT/MIN/MAX/EMA`, `PLANT_SAMPLE_MIN_PX` |
| Coarse jump | `COARSE_SAFETY`, `COARSE_FALLBACK`, `COARSE_UNDERSHOOT_MIN/MAX` |
| Diagnostics | `SERVO_DEBUG` |

## Debug output

With `SERVO_DEBUG = True` (the default), every placement prints:

```
[servo coarse] undershoot=(0.51,0.54) learned_plant=(1.81,1.71)
[servo 01] piece=(420, 633) target=(593, 1387) raw=(+173,+754) step=(+18,+18) finger=(...) plant=(1.81,1.71)
[servo 02] piece=(456, 669) target=(593, 1387) raw=(+137,+718) step=(+18,+18) finger=(...) plant=(1.82,1.74)
...
[auto] servo: ok (locked on piece, 14 iters)
[servo learned] plant=(1.84,1.78) → next coarse undershoot ≈ (0.50,0.52)
```

What to look at when something's off:

| Symptom | Field to inspect |
|---|---|
| Overshoots top row | `plant=` climbs above 2.0 — coarse undershoot will auto-correct |
| Piece never reaches target | `step=(±18, 0)` with `raw` not shrinking → finger overtravel hitting; check Y-axis anti-windup |
| "locked" prints but piece doesn't drop | `PRE_LIFT_MS` too short for this Block Blast build |
| `plant` stuck at `1.50` (init) | sample filter rejecting; lower `PLANT_SAMPLE_MIN_PX` from 4 to 2 |

## Retuning for a new device

The tunables that depend on **device geometry** (and therefore need to
move if you change phone / DPI / resolution):

* `STEP_CLAMP_PX`, `FINE_THRESHOLD_PX`, `LOCK_TOLERANCE_PX` — pixel-
  denominated; scale with capture resolution.
* `FINGER_OVERTRAVEL_Y` — pixel-denominated; bigger phones need
  proportionally more.

The tunables that depend on **Block Blast's behaviour** (move when the
game updates, not when the phone changes):

* `HOLD_MS`, `PRE_LIFT_MS` — gesture acceptance windows.
* `INITIAL_LIFT_PX` — needed to disambiguate hold-vs-flick at grab time.
* `COARSE_SAFETY`, `FINE_SAFETY_FACTOR` — safety margins against the
  plant overshoot character.

The tunables that **self-tune** and don't need manual adjustment:

* `_learned_plant_gx`, `_learned_plant_gy` — measured online.
* `COARSE_FALLBACK` — only used on placement #1; after that the
  measured value drives the coarse jump.

Practical retuning recipe:

1. Bring up the assist GUI with `SERVO_DEBUG = True`.
2. Make 5–10 placements covering the corners and middle of the board.
3. Read the final `[servo learned] plant=(...)` for the device.
4. If that plant gain is well outside `[0.4, 4.0]`, widen the clamps in
   `tunables.py` (`PLANT_GAIN_MIN/MAX`).
5. If the typical placement takes more than 8 closed-loop iterations to
   lock, `COARSE_SAFETY` can be raised toward 1.0 to spend less time in
   the closed loop. If it usually overshoots and reverses, lower it.

Everything else is a knob you should leave alone unless you have a
specific failure mode in front of you. The comments next to each
constant in `tunables.py` are not aspirational — they document real
traces where wrong values caused real problems.
