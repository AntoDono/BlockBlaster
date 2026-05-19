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

Auto-play does **not** rely on a pre-baked finger calibration. Each move
runs a closed-loop visual servo: press the finger on the queue slot,
then loop {template-match the held piece against the board diff,
P-step the finger toward the matched centroid} until both the
centroid error and the match score are inside their tolerances.
Release when locked; lift in place if the loop budget runs out.

The detector is one `cv2.matchTemplate` pass per frame against the
diff of the current `_board_filled_mask` and a pre-grab snapshot of
that same mask. No tracker state, no learning — every frame is a
fresh global search. Every tunable lives at the top of
[`blockblaster/control/servo.py`](../blockblaster/control/servo.py).

Code layout:

- [`blockblaster/control/servo.py`](../blockblaster/control/servo.py) — the entire servo, one file: constants, `_board_filled_mask`, `_make_template`, `_locate_piece`, and the public `place(...) -> bool`.
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
[auto] servo: OK (3 iters)
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
