# Control Stack

[← back to README](../README.md)

Everything device-facing lives in [`blockblaster/control/`](../blockblaster/control/). The rest of the app talks to a single `Device` abstraction and (for auto-play) a scrcpy touch session.

## The `Device` protocol — [`device.py`](../blockblaster/control/device.py)

```python
class Device:
    supports_input: bool
    def start() / stop()
    def get_latest_with_id() -> (frame_bgr, frame_id)   # capture
    def screen_size() -> (w, h)
    def tap(x, y) / swipe(x1,y1,x2,y2, duration_ms)     # input (if supported)
```

`get_latest_with_id` returns the most recent captured frame and a monotonically increasing id — compare the id (not pixels) to detect a new frame. Backends that can't inject input raise `InputNotSupportedError` from `tap`/`swipe`.

`make_device(platform, serial)` picks the backend:

- **`android`** → `AndroidScreenrecordDevice` (fast H.264 `screenrecord` capture), falling back to `AndroidAdbDevice` (`screencap`) if it can't start.
- **`ios`** → `IosReadOnlyDevice` (mirror only).

## Android capture

- [`android_screenrecord.py`](../blockblaster/control/android_screenrecord.py) — streams `adb shell screenrecord` H.264 and decodes frames; high frame rate for the servo loop. Implements `tap`/`swipe` via ADB for simple gestures.
- [`android_adb.py`](../blockblaster/control/android_adb.py) — slower `screencap`-based fallback.
- [`adb_utils.py`](../blockblaster/control/adb_utils.py) — ADB binary / timeout helpers.

## Closed-loop touch — [`scrcpy_control.py`](../blockblaster/control/scrcpy_control.py)

Simple `adb input tap/swipe` can't hold a persistent gesture and react mid-drag, which the [visual servo](visual-servo.md) needs. So auto-play drives touch through **scrcpy-server v1.20**'s wire protocol directly:

- The v1.20 JAR (bundled by the `pyscrcpy` PyPI package) is pushed to the device and launched via `app_process`.
- Connection goes through **adbutils**' direct `localabstract` tunnel rather than `adb forward`, avoiding the race where the host TCP socket connects before the device socket is bound and then closes silently (a real failure mode on Knox/MDM-locked Samsung builds — which is also why v1.20 is used over the hardened 4.x server).
- One video socket (drained, ignored) + one control socket. Touch is a 28-byte big-endian `INJECT_TOUCH_EVENT` packet (`DOWN`/`MOVE`/`UP`).

`ScrcpyControl.open_session()` yields a `ScrcpyTouchSession` exposing `down(x,y) / move(x,y) / up()`. A continuous `DOWN → MOVE… → UP` with arbitrary host-side timing is exactly what the servo's per-frame feedback loop emits. `UP` is sent with pressure/buttons = 0 (the lift transition Block Blast gates its piece-drop on — sending `UP` at full pressure looks like a continued press and the placement is silently ignored). The server is started once per process and cached by serial (failures cached too).

## iOS — [`ios_readonly.py`](../blockblaster/control/ios_readonly.py)

Mirrors a connected iPhone (via tunneld / DVT) for the read-only assist overlay. `supports_input = False` — Apple doesn't allow touch injection without a paired Mac/Xcode signature, so iOS is visualisation only.

## Coordinate helpers — [`coords.py`](../blockblaster/control/coords.py)

Pixel-anchor math (queue slot centres, board cell centres, piece bottom-row anchors) shared by the auto-play/servo path. The servo itself derives its grab point and target footprint from the analyzer's live detections, so these are helpers rather than a hard calibration dependency.

## Other

- [`touch_capture.py`](../blockblaster/control/touch_capture.py) — utility for recording real on-device touch traces (debugging / calibration reference).
