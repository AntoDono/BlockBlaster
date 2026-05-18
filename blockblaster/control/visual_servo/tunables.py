"""Tunable constants and result type for the visual-servo placer.

These are intentionally split out from the main loop so the algorithm
file stays focused on control flow.  Bump these when retuning for a new
device / Block Blast version.

Each constant carries a comment explaining the failure mode that pins
its value — re-read those before changing anything.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Gesture bookends ────────────────────────────────────────────────────
HOLD_MS              = 280   # dwell after DOWN before any MOVE (long-press grab).
                             # Block Blast needs ~250 ms of stationary touch to
                             # register the piece pickup.  Less than that and
                             # the gesture is interpreted as a flick.
PRE_LIFT_MS          = 120   # settle after the final correction before UP, so
                             # the game commits the placement on this frame.
                             # Block Blast occasionally swallows a release
                             # that lands too close to the previous move()
                             # event — keep this generous.
GRAB_Y_NUDGE_PX      = 100   # queue slot_center sits below the icon — press
                             # this far above the center to actually grab.
INITIAL_LIFT_PX      = 80    # frame-px upward nudge AFTER the grab so the
                             # piece pops above the finger (helps Block Blast
                             # confirm the piece is being dragged, not held).

# ── Loop pacing ─────────────────────────────────────────────────────────
MAX_LOOP_S           = 2.5   # total servo budget per placement.  Must
                             # leave room for DOWN/HOLD/UP bookends inside
                             # AUTO_SERVO_BUDGET_MS in app_autoplay.
FRAME_TIMEOUT_S      = 0.6   # how long to wait for a fresh frame per iter.
POST_MOVE_SETTLE_MS  = 100   # sleep after each move() so the next frame
                             # samples a settled piece, not mid-anim.
                             # Bigger = slower glide + more reliable per-iter
                             # observations (Block Blast's drag follower
                             # noticeably lags fast moves).
MAX_NO_PIECE_FRAMES  = 8     # consecutive frames without a detected piece
                             # before giving up.

# ── Lock criteria ───────────────────────────────────────────────────────
STABLE_MATCHES       = 2     # consecutive piece-matches required before lifting.
LOCK_TOLERANCE_PX    = 12    # half-cell-ish tolerance for the "near enough"
                             # lock fallback.  If the detected piece anchor
                             # sits within this many px of the target on both
                             # axes AND has the same cell count as the
                             # suggestion, accept as locked even if the cell
                             # *sets* don't compare equal (the board scanner
                             # flickers ±1 cell at cell-boundary transitions
                             # on a visually-correct drop).

# ── Controller ──────────────────────────────────────────────────────────
STEP_CLAMP_PX        = 18    # max single-step move in frame pixels.  Tight,
                             # because the plant gain (>1) compounds any
                             # finger move; large finger jumps overshoot
                             # wildly.  Also the coarse-mode stride when
                             # |raw_err| > FINE_THRESHOLD_PX, so this
                             # doubles as the "how slowly the piece glides"
                             # knob.
FINE_THRESHOLD_PX    = 30    # |raw_err| at or below this uses the plant-
                             # inverted controller; above it we take fixed
                             # STEP_CLAMP_PX strides directly toward the
                             # target.  Without the two-mode split a small
                             # gain stalls on big errors (gain*err rounds
                             # to 1–2 px / iter) and the loop budget burns
                             # up before reaching far targets.
FINE_SAFETY_FACTOR   = 0.85  # extra de-rating on the inverted-plant
                             # correction so even when the gain estimate
                             # is a touch low we approach the target from
                             # one side rather than overshoot.

# ── Finger ↔ piece geometry ─────────────────────────────────────────────
# Block Blast renders the held piece *above* the finger: the cell of the
# piece that was under the touch-down point stays roughly under the
# finger (with a constant upward "render lift" so the piece isn't
# occluded), and the rest of the piece extends from there.
#
# That means the finger's frame-Y for a piece to land at target_anchor
# depends on the piece's geometry — specifically how far the bottom-row
# centre (which is what ``piece_anchor_px`` returns and ``target_anchor``
# refers to) sits *below* the piece's geometric centre (which is the
# point under the queue-slot grab).  For a 1×N horizontal bar Δrow=0 so
# there's no extra Y bias; for a 4×1 vertical bar Δrow=1.5 and the
# finger needs to be ~180 px *closer* to the target row than the
# horizontal bar would require.
#
# Measured render lift from a successful slot-1 → row-7 placement of
# the 1×4 horizontal bar:
#   finger=(639, 1522) → piece=(659, 1269)   lift = 253
#   finger=(531, 1630) → piece=(360, 1390)   lift = 240
# Use the mean; the closed loop closes whatever residual remains.
FINGER_RENDER_LIFT_PX = 245

# ── Anti-windup ─────────────────────────────────────────────────────────
FINGER_OVERTRAVEL_Y  = 350   # max frame-pixels the finger is allowed to
                             # drift *below* target_anchor.y.  Block Blast
                             # renders the held piece above the finger with
                             # an offset that GROWS as you drag — early in
                             # a placement the gap is ~50 px, late it can
                             # balloon to 300+ px.  The cap has to clear
                             # the largest observed late-drag offset
                             # (≈360 px in one failing trace) but stay
                             # well above the screen edge that caused the
                             # earlier "finger marches off the bottom"
                             # failure (~480 px overtravel).  350 sits in
                             # the safe middle of that window.

# ── Plant-gain estimator ────────────────────────────────────────────────
PLANT_GAIN_INIT      = 1.5   # starting estimate of (piece px / finger px)
                             # before the first motion sample arrives.
PLANT_GAIN_MIN       = 0.4   # clamp on the estimate — finger barely moves
                             # piece (very late drag, near edges).
PLANT_GAIN_MAX       = 2.8   # clamp on the estimate — the "true" smooth-
                             # region ratio observed empirically is ~2.0–2.3.
                             # Higher samples in the log are column-snap
                             # events (finger crosses a cell boundary, piece
                             # jumps a whole column, Δpiece/Δfinger ≈ 10).
                             # If we admit those into the EMA the learned
                             # gain drifts to ~3.5 and the coarse jump under-
                             # shoots by a column or two on the *next* run.
PLANT_GAIN_EMA       = 0.4   # how aggressively to trust a new sample
                             # vs. the running estimate.  Lower = smoother
                             # but slower to adapt.
PLANT_SAMPLE_MIN_PX  = 4     # ignore (Δfinger, Δpiece) samples where the
                             # finger barely moved on that axis — division
                             # noise dominates.
PLANT_SAMPLE_MAX_RATIO = 4.0 # reject samples where dp/df is implausibly
                             # large (column-snap or detection jump).  The
                             # CLAMP above bounds the *estimate*, but a
                             # giant rejected sample shouldn't even pull
                             # toward the clamp ceiling.

# ── Coarse open-loop jump ───────────────────────────────────────────────
COARSE_SAFETY        = 1.0   # multiplier on the inverted-plant coarse jump.
                             # 1.0 = open-loop jump aims for *exactly* the
                             # target.  Drop below 1.0 only when you trust
                             # the closed loop to close an undershoot
                             # (i.e. detection is working consistently).
                             # On devices where the held piece is rendered
                             # outside the calibrated board area, the
                             # closed loop is effectively blind and any
                             # undershoot just becomes a missed placement —
                             # better to land exactly on target on the
                             # open-loop jump and pair this with the
                             # blind-commit fallback in placer.py.
COARSE_FALLBACK      = 0.55  # initial coarse-undershoot fraction used on
                             # the *first* placement of a session, before
                             # the persistent plant-gain estimate has any
                             # samples.  After that the value is ignored
                             # in favour of COARSE_SAFETY / plant_gain.
COARSE_UNDERSHOOT_MIN = 0.20 # bound on the auto-computed fraction so a
COARSE_UNDERSHOOT_MAX = 0.85 # bad early estimate can't produce a
                             # pathological open-loop jump.

# ── Blind commit ────────────────────────────────────────────────────────
# Released-in-place fallback for devices where the scanner can't see the
# held piece during the drag.  Only fires if the finger ended up close
# to where the piece *should* go — otherwise we abort and let the auto-
# loop retry rather than committing a guaranteed misplacement.
BLIND_COMMIT_TOL_PX  = 80    # ≈ ⅔ of a board cell.  If |finger - target|
                             # exceeds this on either axis, the open-loop
                             # jump clearly didn't get us there — abort
                             # instead of lifting in a wrong cell.

# ── Persistence ─────────────────────────────────────────────────────────
# Where per-device learned plant gains are cached on disk.  One JSON file
# per ADB serial, so swapping phones doesn't require relearning.  Edit
# the file by hand to force a value (useful when you've nailed the
# numbers and want them locked in).
#
# Default: ``<repo_root>/learned_device_params`` — lives alongside the
# code so the files travel with the project and can be checked in if
# you want known-good gains shared across machines.  Override with the
# ``BLOCKBLASTER_SERVO_PARAMS_DIR`` env var (e.g. point it at a writable
# location when running from a read-only install).
import os as _os
from pathlib import Path as _Path

# tunables.py lives at <root>/blockblaster/control/visual_servo/tunables.py
# so the project root is three parents up.
_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
PARAMS_DIR = _Path(
    _os.environ.get(
        "BLOCKBLASTER_SERVO_PARAMS_DIR",
        str(_PROJECT_ROOT / "learned_device_params"),
    )
)

# ── Diagnostics ─────────────────────────────────────────────────────────
SERVO_DEBUG = True   # per-iteration ``[servo NN] ...`` prints.  Cheap
                     # and very useful when retuning gains for a new
                     # game version or device.


@dataclass
class ServoResult:
    """Outcome of one ``place_with_servo`` call.

    Returned to callers (``app_autoplay``) so they can log success / abort
    reasons and decide whether to retry.  ``iters`` counts only inside-loop
    iterations — coarse open-loop steps aren't included.
    """
    success: bool
    reason:  str
    iters:   int
