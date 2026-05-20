# Android Auto-Play

[← back to README](../README.md)

End-to-end loop: ADB screen capture → board + queue recognition → value-net
policy → closed-loop visual servo that drags the chosen piece into place on
the device.

- [Choosing a device](#choosing-a-device)
- [Calibrate the grid and queue](#calibrate-the-grid-and-queue)
- [Closed-loop visual servo](#closed-loop-visual-servo)
- [scrcpy server v1.20 + adbutils tunnel](#scrcpy-server-v120--adbutils-tunnel)
- [Run](#run)
- [Control module layout](#control-module-layout)

## Choosing a device

Pick whichever Android target you have.

### Option A — BlueStacks (recommended, easiest to install)

1. Install BlueStacks 5, open it, install Block Blast from the Play Store.
2. Enable ADB: BlueStacks settings → **Advanced** → turn on **Android Debug Bridge**.
3. ADB auto-connect is built into [`play.py`](../play.py) — no manual step
   needed. If it fails, run once manually:
   ```bash
   adb connect 127.0.0.1:5555
   ```

### Option B — Android Studio AVD

1. Android Studio → AVD Manager → create a Pixel 6 (API 33+) device → start it.
2. Install Block Blast via the Play Store inside the AVD.

### Option C — Physical phone (Knox-locked Samsung etc.)

This needs the scrcpy-based touch tunnel described in
[scrcpy server v1.20 + adbutils tunnel](#scrcpy-server-v120--adbutils-tunnel)
below. One-time phone setup:

```
Developer options → "USB debugging"                       ON
Developer options → "USB debugging (Security settings)"   ON   ← Samsung-only, gates INJECT_INPUT_EVENTS
```

Then verify ADB sees the device for any of the three options:

```bash
adb devices   # 127.0.0.1:5555   device   (BlueStacks)
              # emulator-5554    device   (AVD)
              # R58M...XYZ       device   (physical phone)
```

## Calibrate the grid and queue

Once per device / resolution:

```bash
uv run play.py --platform android --mode assist
# Tab = toggle GRID ↔ PIECES mode, drag to set box, R to clear
```

Saved to `assist_config_android.json`. The general assist GUI is documented
in [assist-gui.md](assist-gui.md).

## Closed-loop visual servo

Auto-play does **not** rely on a pre-baked finger calibration. Each
move runs a closed-loop visual servo whose detector is
colour/palette-invariant and whose controller is a PD with a
distance-adaptive step ceiling.

### Detection (frame-diff + template match)

The grayscale board crop is snapshotted just before DOWN as a
baseline; per frame we compute `cv2.absdiff(current, baseline)`,
threshold, morph-close → a binary "this pixel moved" mask. The held
piece is the only thing in motion on the board, so its rendered
footprint is exactly what lights up — regardless of colour,
translucency, or ghost-preview noise. `cv2.matchTemplate` of the
known piece silhouette against that mask gives the rigid initial
pose. No tracker state between frames; every frame is a fresh global
search.

### Per-frame measurement: 5 anchors

Instead of one centroid we sample 5 anchors per frame:

- **TL / TR / BL / BR** — the extreme moving pixel in each corner
  direction of the corresponding extreme cell of the piece silhouette
  (e.g. for TL: topmost-leftmost moving pixel in the top-row leftmost
  cell). Anchored to the silhouette's crisp edges, robust to interior
  mass distribution.
- **C** — centre-of-mass of motion in the most-central cell of the
  silhouette (elbow of an L, middle cell of a line). Robust to
  corner-cell occlusions that would bias corner anchors.

Anchors whose cell coverage falls below `_CELL_MIN_COVERAGE` are
dropped. The controller's error is the mean of `(target − measured)`
over the visible anchors, so partial occlusions (board edges, score
popups) don't bias the average.

### Gesture

1. **DOWN** on the queue slot (nudged up by `GRAB_Y_NUDGE_PX`), held
   `HOLD_MS` so Block Blast registers the long-press grab.
2. **Pre-lift** diagonally to `(board_centre_x, slot_y − INITIAL_LIFT_PX)`
   — centring x prevents wide pieces grabbed from edge queue slots
   from hanging off the board where the matcher can't see them.
3. **Confirm**: wait up to `PRELIFT_CONFIRM_S` for the matcher to
   return a confident detection. Abort if it never confirms — better
   than blindly steering a piece we can't see.
4. **PD loop** until lock or budget elapses.
5. `PRE_LIFT_MS` settle, then **UP**.

### PD controller + adaptive step cap

`P = err / GAIN`, `D = DERIV_GAIN * derr / GAIN`. If D flips the sign
of P, it's nulled — the piece coasts one frame instead of overshooting
near zero error.

The per-iter step ceiling is distance-adaptive:

```
|err| ≥ FAR_ERR_PX   → cap = MAX_STEP_FAR_PX   (cover ground fast)
|err| ≤ NEAR_ERR_PX  → cap = MAX_STEP_NEAR_PX  (fine alignment)
in between           → linearly interpolated
```

Per-axis, so a piece aligned on x but far on y still gets a fast y
step without throwing x off. `MAX_STEP_FAR_PX` is bounded by Block
Blast's drag-follower lag: too fast and the next frame reads a stale
position and the PD overshoots. Each PD step is interpolated into
`MOVE_SUBSTEPS` touch events `MOVE_SUBSTEP_MS` apart so the in-game
drag follower renders continuously.

### Release criteria

UP commits the placement, so the gate is conservative:

```
(tight_lock OR transit_lock)
    AND score ≥ LOCK_SCORE_MIN          # match is real, not a phantom
    AND paired ≥ LOCK_MIN_ANCHORS       # enough anchors to trust error

tight_lock:   |err_x| ≤ LOCK_TOL_PX AND |err_y| ≤ LOCK_TOL_PX
transit_lock: each axis was inside LOCK_TOL_PX at some point this run
              AND is still within 2× tol now (catches diagonal
              pass-through where axes peak on different frames).
```

`LOCK_MIN_ANCHORS` (default 2) lets edge placements release when some
corner anchors are off-board — the mean-of-visible error is still
accurate from as few as 2 anchors.

If the matcher loses the piece for `MAX_NO_PIECE_FRAMES` consecutive
iters, abort. If `MAX_LOOP_S` elapses without lock, lift in place
(don't drag back to the queue).

All tunables live in [`blockblaster/config/params.py`](../blockblaster/config/params.py).

### Code layout

- [`blockblaster/control/servo.py`](../blockblaster/control/servo.py) — the entire servo: `_motion_mask`, `_make_template`, `_piece_anchors`, `_locate_piece`, the PD loop, and the public `place(...) -> bool`.
- [`blockblaster/config/params.py`](../blockblaster/config/params.py) — every servo and autoplay tunable in one file, grouped by subsystem with rationale comments.
- [`blockblaster/control/scrcpy_control.py`](../blockblaster/control/scrcpy_control.py) — host-side scrcpy server lifecycle + INJECT_TOUCH_EVENT packet plumbing (see next section).
- [`scan_board`](../blockblaster/assist/scanner.py) — board state from one frame, used by the recon panel.

## scrcpy server v1.20 + adbutils tunnel

The visual servo needs **per-event** touch control (DOWN → MOVE … → MOVE → UP
as separate packets with arbitrary host-side timing between them). Plain
`adb shell input` is atomic — it owns the gesture from down through up and
can't be paused mid-drag to observe the screen and correct the position.

### What we tried, and why each piece of it matters

The path to working closed-loop control on a non-rooted, Knox-locked
Samsung went through several dead ends. The current solution is the
**union** of the workarounds for each one, so it's worth recording the
chain:

1. **`adb shell input touchscreen draganddrop`** — works on any device but
   atomic: the gesture is owned by the input subsystem from start to
   finish. No way to interleave observation and correction.

2. **`sendevent` over `/dev/input/event*`** — gives per-event control but
   on Knox/MDM-locked Samsung builds the `shell` uid is denied write
   access (`sendevent: /dev/input/event3: Permission denied`). No
   sandbox-friendly workaround on those devices.

3. **scrcpy server (any version)** — runs as `app_process` under the
   `shell` uid, which has the pre-granted `INJECT_INPUT_EVENTS` permission.
   It exposes `IInputManager.injectInputEvent()` over a Unix abstract
   socket. Each incoming control message becomes one `MotionEvent`, so we
   get the per-event control of `sendevent` without needing root.

4. **scrcpy 4.x server** — accepted our connection and then closed the
   socket without sending its handshake byte, with no stderr output. We
   never got a useful diagnostic out of it on the Knox-locked S9.

5. **scrcpy v1.20 server** — predates most of the 4.x hardening and has
   been in continuous use on Android 8+ phones for five years. We get the
   v1.20 JAR from the [`pyscrcpy`](https://pypi.org/project/pyscrcpy/)
   PyPI package (bundled as package data); we do **not** use any of
   pyscrcpy's runtime code — its `ControlSender.touch` is broken (the
   real implementation is commented out and replaced with `adb shell
   input tap`, which loses the per-event property we need). We speak
   v1.20's wire protocol directly in
   [`scrcpy_control.py`](../blockblaster/control/scrcpy_control.py).

6. **`adb forward tcp:LOCAL localabstract:scrcpy` + `socket.create_connection`**
   — still failed with "server closed before sending dummy byte;
   log: (no output)" on both server versions. Root cause: `adb forward`
   accepts the host-side TCP connection immediately and *then* lazily
   tries to reach the device-side abstract socket. If the server's
   `LocalServerSocket.bind()` hasn't completed yet, adb closes the
   host-side socket mute — we observe a successful `connect()` followed
   by an empty `recv()`. No retry logic on our side can recover, because
   from our perspective the connect already succeeded.

7. **`adbutils.create_connection(LOCAL_ABSTRACT, "scrcpy")`** — the actual
   working path. adbutils talks the ADB wire protocol directly
   (`host:tport:<serial>` then `localabstract:scrcpy`); if the abstract
   socket isn't bound yet on the device, the request fails up front with
   `AdbError`, our retry loop spins, and on the next attempt the bind has
   completed and we get a real connection that delivers the dummy byte.

### Setup

`pyscrcpy` is pulled in automatically by `uv sync` (with a uv
`override-dependencies` entry in [`pyproject.toml`](../pyproject.toml) so it
coexists with our newer `av` — pyscrcpy's only `av` import paths are
back-compatible). No separate `brew install scrcpy` is needed; the host's
scrcpy binary, if any, is never invoked.

If you ever want to bump the server version, replace the JAR shipped by
`pyscrcpy` and update `_SCRCPY_VERSION` + `_PACKFMT` in
[`scrcpy_control.py`](../blockblaster/control/scrcpy_control.py) to match
that version's wire protocol.

### Wire protocol (v1.20)

After tunnel setup the host opens **two** sockets to `localabstract:scrcpy`:

| # | Direction | Purpose |
|---|-----------|---------|
| 1 | device → host | Video. Server writes one dummy byte (handshake), 64-byte device name, 4-byte `(width, height)` resolution, then an H.264 stream. We drain it on a background thread so the socket buffer never blocks — capture stays on our existing `screenrecord` pipeline. |
| 2 | host → device | Control. Each `INJECT_TOUCH_EVENT` packet drives one `MotionEvent`. |

INJECT_TOUCH_EVENT packet (28 bytes, big-endian):

```
u8   type           = 2 (TYPE_INJECT_TOUCH_EVENT)
u8   action         = 0 (DOWN) / 1 (UP) / 2 (MOVE)
i64  pointer_id     = 0      (non-negative → SOURCE_TOUCHSCREEN finger)
i32  x
i32  y
u16  screen_w
u16  screen_h
u16  pressure       = 0xFFFF  (fixed-point 1.0)
i32  buttons        = 1       (primary)
```

A continuous DOWN → MOVE… → UP gesture is just a sequence of those packets
with arbitrary host-side timing between them — which is exactly what the
visual servo needs to react to ghost-preview feedback mid-drag.

## Run

```bash
uv run play.py --platform android                         # headless
uv run play.py --platform android --display               # with pygame preview
uv run play.py --platform android --serial emulator-5554  # explicit serial
```

| Flag | Default | Description |
|------|---------|-------------|
| `--platform` | required | `ios` or `android` |
| `--mode` | `auto` on android, `assist` on ios | `assist` or `auto` |
| `--display` | off | Pygame preview window in auto mode |
| `--serial` | auto-detect | ADB device serial |

On a successful run you should see:

```
[scrcpy] v1.20 control ready (device='SM-G960U', screen=1080x2220)
[auto] 2x2 slot=2 → row=4 col=4  servo (541, 1626) → (533, 1027)  …
[servo] pre-lift confirmed (score=0.92, anchors=5/5)
[servo 1] err=(+12,-48) derr=(+0,+0) score=0.94 anchors=5/5 step=(+8,-32) finger=(549, 1594)
[servo 4] LOCK[TIGHT] err=(-3,+2) best=(3,1) score=0.98 anchors=5/5
[auto] servo: ok
```

If startup fails, captured server logs are printed under `[scrcpy] …` lines
in real time (no buffering) so silent crashes become visible without
re-running.

## Control module layout

```
blockblaster/control/
  device.py          # Device base class + InputNotSupportedError + make_device()
  android_adb.py     # AndroidAdbDevice  (screencap + swipe via adb)
  android_screenrecord.py  # AndroidScreenRecordDevice (H.264 live capture)
  ios_readonly.py    # IosReadOnlyDevice (wraps DeviceStream, no input)
  adb_utils.py       # ADB binary path + timeout constants
  coords.py          # slot_center_px / cell_center_px / piece_anchor_px
  scrcpy_control.py  # ScrcpyControl: runs scrcpy-server v1.20 on-device,
                     # tunnels INJECT_TOUCH_EVENT via adbutils LOCAL_ABSTRACT
  servo.py           # place(): closed-loop template-match placement
  touch_capture.py   # TouchCapture: getevent stream (kept for diagnostics)
  auto_player.py     # main loop: capture → scan → advise → servo → wait
```
