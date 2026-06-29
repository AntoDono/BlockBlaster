# Visual Servo

[← back to README](../README.md)

[`control/servo.py → place()`](../blockblaster/control/servo.py) executes one advisor suggestion on the device: press the tray piece, drag it onto the target cells under continuous visual feedback, and release once it's locked on target. All tunables are module constants at the top of `servo.py`.

```
DOWN on tray piece ─▶ lift ─▶ [ track piece · PD-step finger ]loop─▶ LOCK ─▶ UP
```

## The gesture

Driven by a persistent **scrcpy v1.20 touch session** ([control-stack.md](control-stack.md)) so a single DOWN→MOVE…→UP gesture can react to feedback mid-drag:

1. **Grab** — `DOWN` on the detected tray piece's bbox centre, hold `HOLD_MS`.
2. **Lift** — an initial upward nudge (`INITIAL_LIFT_PX`) so the piece pops above the finger, with a random `±START_NOISE_X_PX` x-jitter so a retry doesn't deterministically repeat a failed path.
3. **Servo loop** — until locked or the `MAX_LOOP_S` budget elapses.
4. **Release** — `UP` after a short `PRE_LIFT_MS` settle.

Frame coordinates are scaled to device pixels per packet.

## Tracking the held piece (edges, not centroids)

Detection is colour/palette-invariant. Just before `DOWN`, a grayscale **baseline** of the observed region is cached. Each frame:

1. `absdiff(current, baseline)` → threshold → morph-close = a binary **motion mask** (the moving piece's edges light up).
2. Take the **largest connected component** (the heavy, concentrated blob = the piece), reject it if its area is below `PIECE_AREA_FRAC` of the piece's expected footprint (scales the noise floor to the piece, so flicker can't masquerade as it).
3. The piece's position is the **percentile-trimmed bounding box (extent)** of that component — *edges*, not a centre-of-mass, which was unstable because the frame-diff lights up edges.

The observed region is the board **expanded toward the screen edges** (`OBSERVE_MARGIN_PX`), so a piece near a board border isn't clipped. Board cell math stays on the true board bbox.

## The controller

The error is a **5-point correspondence**: centre + 4 corners of the measured bbox mapped to the target footprint's 5 points, averaged. A PD law turns it into a finger step:

```
corr = clamp( (err + DERIV_GAIN · Δerr) / GAIN , ±cap )
```

The D term damps overshoot; each step is interpolated into `MOVE_SUBSTEPS` touch-moves for a smooth on-device drag. **Release (LOCK)** when both axes of the error are within `LOCK_TOL_PX` *and* the template-match score ≥ `LOCK_SCORE_MIN`.

## Two-phase focus

Gated on the **piece's** distance to target (not the finger — which floats below the piece by the render lift):

- **Traveling** (far): full-board motion mask, coarse `MAX_STEP_PX` steps. The piece is tracked anywhere it goes.
- **Focused** (within `APPROACH_RADIUS_PX`): the search narrows to a **local window around the target** (`ROI_MARGIN_PX`) intersected with **cells that were empty at baseline** — this drops a near-complete row's *glow* (which lights up already-filled cells) so it can't be mistaken for the piece. Steps tighten to `FINE_STEP_PX` for a careful approach. Strictly empty-masked here, no fallback.

The empty-cell mask is derived adaptively (Otsu split of the baseline into dark/empty vs bright/filled), so it self-tunes to the board palette.

## Containment

If the piece's bbox **centre** drifts outside the board, the step is overridden to drive it back inward at full `MAX_STEP_PX` (status: *BOUNDARY HIT — PUSHING BACK*). Using the centre — not the edges — avoids false triggers from the drag preview rendering larger than the footprint and from legitimate edge placements.

## Status events

Each iteration the servo publishes a `ServoDebug` (target/measured boxes + 5 points, finger, err, corr, score, observed region, and a status string) consumed by the Frame Diff debug overlay ([assist-gui.md](assist-gui.md)). Statuses: `TRAVELING`, `FOCUSED — FINE APPROACH`, `BOUNDARY HIT — PUSHING BACK`, `SEARCHING FOR PIECE…`, `LOCKED — RELEASING`.

## Tuning cheatsheet

| Symptom | Knob |
|---------|------|
| Overshoots near target | lower `FINE_STEP_PX` / raise `DERIV_GAIN` |
| Too slow to arrive | raise `MAX_STEP_PX` / lower `GAIN` |
| Releases off-target | lower `LOCK_TOL_PX` |
| Won't lock (jitter) | raise `LOCK_TOL_PX` / lower `LOCK_SCORE_MIN` |
| Chases noise | raise `PIECE_AREA_FRAC` / `DIFF_THRESHOLD` |
| Glow mistaken for piece | shrink `APPROACH_RADIUS_PX` reliance / verify Otsu split |
| Focus window clipped at edge | raise `OBSERVE_MARGIN_PX` |
