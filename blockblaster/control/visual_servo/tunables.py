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
PLANT_GAIN_MAX       = 4.0   # clamp on the estimate — small finger
                             # nudges fling the piece (high-offset region).
PLANT_GAIN_EMA       = 0.4   # how aggressively to trust a new sample
                             # vs. the running estimate.  Lower = smoother
                             # but slower to adapt.
PLANT_SAMPLE_MIN_PX  = 4     # ignore (Δfinger, Δpiece) samples where the
                             # finger barely moved on that axis — division
                             # noise dominates.

# ── Coarse open-loop jump ───────────────────────────────────────────────
COARSE_SAFETY        = 0.92  # safety multiplier on the inverted-plant
                             # coarse jump.  We aim for the open-loop jump
                             # to land at COARSE_SAFETY × target so the
                             # piece is always *just short* of the
                             # destination — the closed loop closes an
                             # undershoot easily; an overshoot requires
                             # reversing direction and is harder to learn
                             # from (sticky frames pollute the estimator).
COARSE_FALLBACK      = 0.55  # initial coarse-undershoot fraction used on
                             # the *first* placement of a session, before
                             # the persistent plant-gain estimate has any
                             # samples.  After that the value is ignored
                             # in favour of COARSE_SAFETY / plant_gain.
COARSE_UNDERSHOOT_MIN = 0.20 # bound on the auto-computed fraction so a
COARSE_UNDERSHOOT_MAX = 0.85 # bad early estimate can't produce a
                             # pathological open-loop jump.

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
