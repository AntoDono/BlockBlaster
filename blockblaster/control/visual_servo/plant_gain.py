"""Online plant-gain estimator + persistent learned cache.

The "plant gain" is the per-axis ratio ``piece_px / finger_px`` — i.e.
how many board pixels the held piece moves for every device pixel we
slide the finger.  It is approximately constant for a given device /
game version, but unknown ahead of time, and drifts a little with drag
distance.

We estimate it in two layers:

* **Per-call EMA** (``update_sample``): every loop iteration that
  observed a valid finger / piece motion contributes one sample to a
  smoothed running estimate held on the stack of ``place_with_servo``.
* **Persistent cache**: when a placement ends, the per-call estimate is
  copied into a module-scope value *and* written to disk under
  :data:`tunables.PARAMS_DIR` keyed by the device serial.  The next
  process invocation (or the next placement on the same device) reads
  it back, so the open-loop coarse jump can be sized correctly on the
  very first move instead of falling back to the initial guess.

The on-disk file is small JSON — edit it by hand to lock in known-good
gains, or delete it to force relearning.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from blockblaster.control.visual_servo.tunables import (
    COARSE_FALLBACK,
    COARSE_SAFETY,
    COARSE_UNDERSHOOT_MAX,
    COARSE_UNDERSHOOT_MIN,
    PARAMS_DIR,
    PLANT_GAIN_EMA,
    PLANT_GAIN_INIT,
    PLANT_GAIN_MAX,
    PLANT_GAIN_MIN,
    PLANT_SAMPLE_MAX_RATIO,
    PLANT_SAMPLE_MIN_PX,
)

# ── Persistent learned state (in-memory mirror of the on-disk file) ────
_learned_plant_gx: Optional[float] = None
_learned_plant_gy: Optional[float] = None
_active_device:    Optional[str]   = None


def _safe_filename(device_id: str) -> str:
    """Sanitise an ADB serial so it's safe as a filename component."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in device_id)


def _params_path(device_id: str) -> Path:
    return PARAMS_DIR / f"{_safe_filename(device_id)}.json"


def bind_device(device_id: str) -> None:
    """Switch the active device and load its cached gains from disk.

    Called once at the start of every ``place_with_servo`` invocation
    so the same Python process can drive multiple devices and keep
    their learned plant gains separate.
    """
    global _learned_plant_gx, _learned_plant_gy, _active_device
    if device_id == _active_device:
        return
    _active_device = device_id
    path = _params_path(device_id)
    if not path.is_file():
        _learned_plant_gx = None
        _learned_plant_gy = None
        return
    try:
        data = json.loads(path.read_text())
        gx = data.get("plant_gx")
        gy = data.get("plant_gy")
        _learned_plant_gx = float(gx) if gx is not None else None
        _learned_plant_gy = float(gy) if gy is not None else None
    except (OSError, ValueError, json.JSONDecodeError):
        _learned_plant_gx = None
        _learned_plant_gy = None


def get_learned() -> tuple[Optional[float], Optional[float]]:
    """Return the cached per-axis plant gains, or ``(None, None)``."""
    return _learned_plant_gx, _learned_plant_gy


def set_learned(gx: float, gy: float, *, persist: bool = True) -> None:
    """Overwrite the cache with new estimates and write to disk.

    Pass ``persist=False`` to update only the in-memory mirror (useful
    for tests or when you don't want a placement run to overwrite a
    hand-tuned file).
    """
    global _learned_plant_gx, _learned_plant_gy
    _learned_plant_gx = gx
    _learned_plant_gy = gy
    if persist and _active_device is not None:
        _write_to_disk()


def reset_learned() -> None:
    """Forget what we've learned for the active device (in-memory + disk)."""
    global _learned_plant_gx, _learned_plant_gy
    _learned_plant_gx = None
    _learned_plant_gy = None
    if _active_device is not None:
        path = _params_path(_active_device)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _write_to_disk() -> None:
    """Atomically write the current in-memory values for the active device."""
    if _active_device is None:
        return
    path = _params_path(_active_device)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device": _active_device,
            "plant_gx": _learned_plant_gx,
            "plant_gy": _learned_plant_gy,
            "updated_at": time.time(),
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)
    except OSError:
        pass


def seed_estimates() -> tuple[float, float]:
    """Starting per-axis estimates for a fresh ``place_with_servo`` call.

    Returns the learned cache values if any exist, otherwise the bare
    ``PLANT_GAIN_INIT`` guess.  The caller then updates these in-place
    via :func:`update_sample` for each loop iteration.
    """
    gx, gy = get_learned()
    return (
        gx if gx is not None else PLANT_GAIN_INIT,
        gy if gy is not None else PLANT_GAIN_INIT,
    )


def update_sample(plant_g: float, df: int, dp: int) -> float:
    """EMA-update a per-axis plant-gain estimate with one motion sample.

    Filters that protect the estimate:

    * ``|df| >= PLANT_SAMPLE_MIN_PX`` — finger must have moved enough that
      the ratio isn't dominated by detection noise.
    * ``df * dp > 0`` — piece must have moved in the *same direction* as
      the finger; sticky frames and board-edge clipping show up as
      opposite-sign or zero ``dp`` and we ignore them.
    * Sample clamped to ``[PLANT_GAIN_MIN, PLANT_GAIN_MAX]`` before EMA so
      one outlier can't pull the estimate to absurd values.

    Returns the updated estimate (unchanged if the sample was rejected).
    """
    if abs(df) < PLANT_SAMPLE_MIN_PX:
        return plant_g
    if df * dp <= 0:
        return plant_g
    sample = dp / df
    # Column-snap rejection: when the finger crosses a cell boundary the
    # piece jumps a whole column, producing dp/df ratios of 8–13.  If we
    # admit those into the EMA the estimate drifts to the clamp ceiling
    # and the next placement's coarse jump undershoots.  Throw the
    # sample away rather than letting it pull the estimate.
    if abs(sample) > PLANT_SAMPLE_MAX_RATIO:
        return plant_g
    sample = max(PLANT_GAIN_MIN, min(PLANT_GAIN_MAX, sample))
    return (1 - PLANT_GAIN_EMA) * plant_g + PLANT_GAIN_EMA * sample


def coarse_undershoot_for(plant_g: Optional[float]) -> float:
    """Pick the open-loop coarse-jump fraction for one axis.

    Without a learned plant-gain we fall back to ``COARSE_FALLBACK``.
    With one, the ideal jump is ``1 / plant`` (lands the piece exactly
    on target) — we de-rate by ``COARSE_SAFETY`` so we always undershoot
    slightly, then clamp to ``[COARSE_UNDERSHOOT_MIN, COARSE_UNDERSHOOT_MAX]``
    so an early-session outlier can't produce a pathological jump.
    """
    if plant_g is None or plant_g <= 0:
        return COARSE_FALLBACK
    raw = COARSE_SAFETY / plant_g
    return max(COARSE_UNDERSHOOT_MIN, min(COARSE_UNDERSHOOT_MAX, raw))
