"""Closed-loop touch control via scrcpy-server **v1.20**.

Why v1.20 and not the modern (4.x) protocol?
--------------------------------------------
scrcpy 4.x uses ``scid=`` / key=value launch args, a single multiplexed
abstract socket per stream, and a 32-byte INJECT_TOUCH_EVENT packet.
On Knox/MDM-locked Samsung builds (e.g. SM-G960U on US carriers) the
4.x server starts, accepts our connection, then closes the socket
without sending its handshake byte and without writing anything to
stderr — leaving us with no way to diagnose what tripped its safety
check.  v1.20 predates most of that hardening, has been in continuous
use for five years, and is known to work on Android 8+ where 4.x
silently dies.

We get the v1.20 JAR from the ``pyscrcpy`` package on PyPI (it bundles
``scrcpy-server.jar`` 1.20 verbatim).  We do **not** use any of
pyscrcpy's runtime code — its ``ControlSender.touch`` is broken (the
real implementation is commented out and replaced with
``adb shell input tap`` which loses the persistent-gesture property
that closed-loop servoing requires).  We speak v1.20's wire protocol
directly here.

Wire protocol (v1.20)
---------------------
1. Push ``scrcpy-server.jar`` to ``/data/local/tmp/``.
2. ``adb forward tcp:LOCAL localabstract:scrcpy``.
3. ``adb shell CLASSPATH=… app_process / com.genymobile.scrcpy.Server
   <positional args>`` — ``tunnel_forward=true`` makes the server
   listen on the abstract socket and our forward connects in.
4. Open **two** sockets to the same forward:
     • first  = video  (server writes 1 dummy byte + 64-byte device
                        name + 4-byte resolution, then an H.264
                        stream we drain into /dev/null);
     • second = control (we send INJECT_TOUCH_EVENT packets here).
5. INJECT_TOUCH_EVENT, 28 bytes big-endian::

       u8   type           = 2
       u8   action         = 0 (DOWN) / 1 (UP) / 2 (MOVE)
       i64  pointer_id     = 0   (non-negative → SOURCE_TOUCHSCREEN)
       i32  x
       i32  y
       u16  screen_w
       u16  screen_h
       u16  pressure       = 0xFFFF  (fixed-point 1.0)
       i32  buttons        = 1

A continuous DOWN → MOVE… → UP gesture is just a sequence of those
packets with arbitrary host-side timing between them — which is what
the visual servo needs to react to ghost-preview feedback mid-drag.
"""

from __future__ import annotations

import os
import random
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

from blockblaster.control.adb_utils import ADB_BIN, ADB_TIMEOUT

# adbutils gives us a direct LOCAL_ABSTRACT connection through the ADB
# protocol — bypassing the `adb forward` race where adb cheerfully
# accepts the host-side TCP socket before the device-side abstract
# socket is bound and then silently closes it.  That race is the root
# cause of "server closed before sending dummy byte; log: (no output)"
# we observed on the Knox-locked S9.
from adbutils import adb as _adbutils_adb
from adbutils import AdbError, Network as _AdbNetwork

# v1.20's BuildConfig.VERSION_NAME.  The server cross-checks this and
# refuses to start on a mismatch, so it must match the JAR we push.
_SCRCPY_VERSION = "1.20"
_DEVICE_JAR_PATH = "/data/local/tmp/scrcpy-server.jar"

# MotionEvent actions (android.view.MotionEvent)
ACTION_DOWN = 0
ACTION_UP   = 1
ACTION_MOVE = 2

# v1.20 routes (pointer_id, buttons) onto either SOURCE_TOUCHSCREEN or
# SOURCE_MOUSE.  buttons != 0 + pointer_id == -1 → mouse.  Anything
# else → finger.  pointer_id 0 with buttons set is the canonical
# "primary finger" pattern scrcpy's own client uses.
_POINTER_ID_FINGER = 0
_PRESSURE_FULL_FP16 = 0xFFFF       # u16 fixed-point 1.0
_PRIMARY_BUTTON     = 1

_TYPE_INJECT_TOUCH_EVENT = 2

# struct: type(u8) action(u8) pointer_id(i64) x(i32) y(i32)
#         screen_w(u16) screen_h(u16) pressure(u16) buttons(i32)
# → 28 bytes, big-endian.  Matches v1.20 ControlMessageReader.
_PACKFMT = "!BBqiiHHHi"
assert struct.calcsize(_PACKFMT) == 28


# ---------------------------------------------------------------------------
# scrcpy-server.jar locator — pyscrcpy bundles v1.20 in its package data.
# ---------------------------------------------------------------------------

def _find_server_jar() -> str:
    """Return the absolute path to scrcpy-server.jar v1.20.

    We rely on ``pyscrcpy``'s bundled copy so the JAR version is
    pinned alongside our protocol implementation — no drift between
    what we speak and what the device runs.
    """
    try:
        import pyscrcpy
    except ImportError as exc:
        raise RuntimeError(
            "pyscrcpy is required for the v1.20 scrcpy-server JAR. "
            "Run `uv sync`."
        ) from exc
    pkg_dir = os.path.dirname(pyscrcpy.__file__)
    jar = os.path.join(pkg_dir, "scrcpy-server.jar")
    if not os.path.isfile(jar):
        raise FileNotFoundError(
            f"Expected v1.20 JAR at {jar} (shipped by pyscrcpy)."
        )
    return jar


# ---------------------------------------------------------------------------
# Persistent server + control socket
# ---------------------------------------------------------------------------

class ScrcpyControl:
    """One scrcpy-server v1.20 instance + control socket, reused across drags.

    Server startup costs ~500–800 ms.  Reusing it across the lifetime
    of a play session means each gesture pays only the ~100 µs TCP
    write cost per MotionEvent.

    Not thread-safe externally — serialize via the session API.
    """

    def __init__(self, serial: str, screen_w: int, screen_h: int) -> None:
        self._serial = serial
        # Screen dims travel in every packet's header.  Servoing
        # already feeds us frame-space ⇒ device-space mapped coords,
        # so this is just metadata for scrcpy's PositionMapper.
        self._sw     = screen_w
        self._sh     = screen_h
        # adbutils handles the tunnel; no host-side TCP port needed.
        self._adb_dev = _adbutils_adb.device(serial=serial)
        self._video: Optional[socket.socket] = None  # drained, ignored
        self._ctrl:  Optional[socket.socket] = None
        self._proc:  Optional[subprocess.Popen] = None
        self._lock   = threading.Lock()
        # Server log lines + drainer threads.  Drained continuously so
        # the OS pipe / socket buffers never fill.
        self._log_buf:   list[str] = []
        self._log_lock   = threading.Lock()
        self._log_thread:   Optional[threading.Thread] = None
        self._video_thread: Optional[threading.Thread] = None
        self._alive = False
        self._start()

    # ── Startup / teardown ────────────────────────────────────────────

    def _adb(self, *args: str, timeout: float = ADB_TIMEOUT) -> str:
        result = subprocess.run(
            [ADB_BIN, "-s", self._serial, *args],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"adb {' '.join(args)} failed (rc={result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()!r}"
            )
        return result.stdout

    def _start(self) -> None:
        jar_host = _find_server_jar()
        self._adb("push", jar_host, _DEVICE_JAR_PATH)

        # v1.20's abstract socket name is the literal "scrcpy" (no scid).
        # Only one server can run at a time per device — fine for us.
        # We deliberately do NOT use `adb forward`; adbutils tunnels the
        # connection directly through the ADB protocol so connect() only
        # succeeds once the device-side LocalServerSocket is actually
        # bound (avoiding the silent-close race we hit before).

        # v1.20 positional launch args, in order:
        #   version, log_level, max_size, bitrate, max_fps,
        #   lock_screen_orientation, tunnel_forward, crop,
        #   send_frame_rate, control, display_id, show_touches,
        #   stay_awake, codec_options, encoder_name, power_off_on_close
        positional = [
            _SCRCPY_VERSION,
            "info",
            "0",          # max_size (no clamp)
            "8000000",    # bitrate (irrelevant; we drain video)
            "0",          # max_fps (server default)
            "-1",         # lock_screen_orientation = UNLOCKED
            "true",       # tunnel_forward  (server listens, we connect)
            "-",          # crop (none)
            "false",      # send_frame_rate
            "true",       # control ENABLED
            "0",          # display_id
            "false",      # show_touches
            "true",       # stay_awake
            "-",          # codec_options
            "-",          # encoder_name
            "false",      # power_off_screen_on_close
        ]
        server_cmd = (
            f"CLASSPATH={_DEVICE_JAR_PATH} app_process / "
            f"com.genymobile.scrcpy.Server " + " ".join(positional)
        )
        self._proc = subprocess.Popen(
            [ADB_BIN, "-s", self._serial, "shell", server_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self._log_thread = threading.Thread(
            target=self._drain_logs, name="scrcpy-log", daemon=True,
        )
        self._log_thread.start()

        # ── Open the VIDEO socket first; v1.20 sends the dummy byte
        # here, plus device-name + resolution metadata.
        self._video = self._connect_with_retry(deadline=time.monotonic() + 6.0)

        # Dummy byte (server writes \x00 immediately after accept).
        self._video.settimeout(3.0)
        try:
            dummy = self._video.recv(1)
        finally:
            self._video.settimeout(None)
        if not dummy:
            time.sleep(0.3)            # let log drainer scoop final bytes
            log = self._collect_log()
            self.close()
            raise RuntimeError(
                f"scrcpy v1.20 server closed before sending dummy byte; "
                f"log:\n{log}"
            )

        # ── Now the CONTROL socket (second accept).
        self._ctrl = self._connect_with_retry(deadline=time.monotonic() + 2.0)
        self._ctrl.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        # ── Device-name (64 bytes) + resolution (u16 w, u16 h) on video.
        self._video.settimeout(3.0)
        try:
            name_raw = self._recv_exact(self._video, 64)
            res_raw  = self._recv_exact(self._video, 4)
        finally:
            self._video.settimeout(None)
        device_name = name_raw.decode("utf-8", "replace").rstrip("\x00")
        dev_w, dev_h = struct.unpack(">HH", res_raw)

        # If the device reports a different live screen size than what
        # the caller passed in (e.g. the user rotated mid-session),
        # trust the device — our PositionMapper rejects out-of-bounds.
        if (dev_w, dev_h) != (self._sw, self._sh):
            print(
                f"[scrcpy] device reports {dev_w}x{dev_h}, "
                f"caller said {self._sw}x{self._sh} — using device values."
            )
            self._sw, self._sh = dev_w, dev_h

        # ── Drain the video stream forever so the server isn't blocked
        # writing to a full TCP buffer.  We don't decode — that's what
        # screenrecord is for.
        self._alive = True
        self._video_thread = threading.Thread(
            target=self._drain_video, name="scrcpy-video", daemon=True,
        )
        self._video_thread.start()

        print(
            f"[scrcpy] v{_SCRCPY_VERSION} control ready "
            f"(device={device_name!r}, screen={self._sw}x{self._sh})"
        )

    def _connect_with_retry(self, deadline: float) -> socket.socket:
        """Open a fresh socket to ``localabstract:scrcpy`` via adbutils.

        adbutils talks the ADB wire protocol directly (``host:tport:`` +
        ``localabstract:scrcpy``) — if the abstract socket isn't bound
        yet on the device, the request fails fast with ``AdbError`` and
        we retry, instead of the ``adb forward`` failure mode where a
        host-side TCP socket appears successful and then closes mute.
        """
        last_exc: Optional[Exception] = None
        while time.monotonic() < deadline:
            assert self._proc is not None
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"scrcpy server exited before accept "
                    f"(rc={self._proc.returncode}); log:\n{self._collect_log()}"
                )
            try:
                conn = self._adb_dev.create_connection(
                    _AdbNetwork.LOCAL_ABSTRACT, "scrcpy",
                )
                # adbutils returns an AdbConnection that quacks like a
                # socket via .conn — unwrap to a real socket so we can
                # use settimeout/sendall/recv unmodified.
                sock = conn.conn if hasattr(conn, "conn") else conn
                if not isinstance(sock, socket.socket):
                    raise RuntimeError(
                        f"adbutils returned {type(sock).__name__}, "
                        "expected socket.socket"
                    )
                return sock
            except (AdbError, OSError) as exc:
                last_exc = exc
                time.sleep(0.1)
        raise RuntimeError(
            f"could not connect to scrcpy v1.20 server: {last_exc}; "
            f"log:\n{self._collect_log()}"
        )

    @staticmethod
    def _recv_exact(sock: socket.socket, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise RuntimeError("scrcpy socket closed mid-handshake")
            buf.extend(chunk)
        return bytes(buf)

    def _drain_video(self) -> None:
        """Continuously read+discard H.264 bytes so the server isn't
        backpressured.  We never decode — screenrecord remains our
        capture path."""
        sock = self._video
        if sock is None:
            return
        sock.settimeout(1.0)
        try:
            while self._alive:
                try:
                    data = sock.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
        except Exception:
            pass

    def _kill_proc(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def _drain_logs(self) -> None:
        """Continuously read server stdout into ``self._log_buf``.

        Prevents the OS pipe buffer (~64 KB on macOS) from filling and
        blocking the server when it logs anything; also feeds our
        failure diagnostics.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in iter(proc.stdout.readline, b""):
                text = line.decode(errors="replace").rstrip()
                if not text:
                    continue
                with self._log_lock:
                    self._log_buf.append(text)
                # Surface server logs in real time so silent crashes
                # become visible without re-running.
                print(f"[scrcpy] {text}", file=sys.stderr)
        except Exception:
            pass

    def _collect_log(self) -> str:
        with self._log_lock:
            return "\n".join(self._log_buf[-40:]) or "(no output)"

    def close(self) -> None:
        self._alive = False
        for s in (self._ctrl, self._video):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
        self._ctrl = None
        self._video = None
        self._kill_proc()

    # ── Public touch API ──────────────────────────────────────────────

    def send_touch(self, action: int, x: int, y: int) -> None:
        """Send one ``INJECT_TOUCH_EVENT`` packet on the control socket.

        Coordinates are clamped to the screen so a tiny overshoot from
        the visual servo doesn't get silently rejected.
        """
        cx = max(0, min(self._sw - 1, int(x)))
        cy = max(0, min(self._sh - 1, int(y)))
        packet = struct.pack(
            _PACKFMT,
            _TYPE_INJECT_TOUCH_EVENT,
            action,
            _POINTER_ID_FINGER,
            cx, cy,
            self._sw, self._sh,
            _PRESSURE_FULL_FP16,
            _PRIMARY_BUTTON,
        )
        with self._lock:
            if self._ctrl is None:
                raise RuntimeError("scrcpy control socket closed")
            self._ctrl.sendall(packet)

    def open_session(self) -> "ScrcpyTouchSession":
        return ScrcpyTouchSession(self)


class ScrcpyTouchSession:
    """One DOWN → MOVE… → UP gesture over a shared :class:`ScrcpyControl`.

    Mirrors the old ``SendeventSession`` API exactly so ``visual_servo``
    can swap backends without changing its control flow.
    """

    def __init__(self, parent: ScrcpyControl) -> None:
        self._parent = parent
        self._down   = False
        self._closed = False
        # Last known finger position, replayed in the UP packet (Android
        # uses these coords as the touch-up location).
        self._last_xy: tuple[int, int] = (0, 0)

    def down(self, x: int, y: int) -> None:
        if self._down:
            raise RuntimeError("ScrcpyTouchSession: down() called twice")
        self._parent.send_touch(ACTION_DOWN, x, y)
        self._last_xy = (x, y)
        self._down = True

    def move(self, x: int, y: int) -> None:
        if not self._down:
            raise RuntimeError("ScrcpyTouchSession: move() before down()")
        self._parent.send_touch(ACTION_MOVE, x, y)
        self._last_xy = (x, y)

    def up(self) -> None:
        if not self._down:
            return
        x, y = self._last_xy
        self._parent.send_touch(ACTION_UP, x, y)
        self._down = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._down:
                self.up()
        except Exception:
            pass

    def __enter__(self) -> "ScrcpyTouchSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Module-level cache so each Device only starts the server once per process.
# ---------------------------------------------------------------------------

_cache: dict[str, Optional[ScrcpyControl]] = {}
_cache_lock = threading.Lock()


def get_scrcpy(
    serial: str, screen_w: int, screen_h: int,
) -> Optional[ScrcpyControl]:
    """Return a cached :class:`ScrcpyControl`, or ``None`` on failure.

    Failure is cached too — startup is heavy and a Knox/permission
    block is permanent within a session.
    """
    with _cache_lock:
        if serial in _cache:
            return _cache[serial]
        try:
            c: Optional[ScrcpyControl] = ScrcpyControl(serial, screen_w, screen_h)
        except Exception as exc:
            print(f"[scrcpy] control unavailable: {exc}")
            c = None
        _cache[serial] = c
        return c
