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
* **Persistent cache** (module-scope ``_learned_plant_g{x,y}``): when a
  placement ends, we copy the per-call estimate into module variables so
  the *next* placement can seed both its open-loop coarse jump and its
  loop-side estimate from real data instead of the bare initial guess.

The cache is process-local — restart the GUI and it resets to ``None``.
"""

from __future__ import annotations

from typing import Optional

from blockblaster.control.visual_servo.tunables import (
    COARSE_FALLBACK,
    COARSE_SAFETY,
    COARSE_UNDERSHOOT_MAX,
    COARSE_UNDERSHOOT_MIN,
    PLANT_GAIN_EMA,
    PLANT_GAIN_INIT,
    PLANT_GAIN_MAX,
    PLANT_GAIN_MIN,
    PLANT_SAMPLE_MIN_PX,
)

# ── Persistent learned state ────────────────────────────────────────────
# Module-scope: survives across ``place_with_servo`` calls within the
# same Python process, gets re-initialised on import / process restart.
_learned_plant_gx: Optional[float] = None
_learned_plant_gy: Optional[float] = None


def get_learned() -> tuple[Optional[float], Optional[float]]:
    """Return the cached per-axis plant gains, or ``(None, None)``."""
    return _learned_plant_gx, _learned_plant_gy


def set_learned(gx: float, gy: float) -> None:
    """Overwrite the cache with new estimates (called at end of placement)."""
    global _learned_plant_gx, _learned_plant_gy
    _learned_plant_gx = gx
    _learned_plant_gy = gy


def reset_learned() -> None:
    """Forget what we've learned (e.g. when switching devices mid-session)."""
    global _learned_plant_gx, _learned_plant_gy
    _learned_plant_gx = None
    _learned_plant_gy = None


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
