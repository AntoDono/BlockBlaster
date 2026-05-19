"""All auto-play and visual-servo tunables, in one place.

Organised by subsystem.  Re-read the comment next to each knob before
changing it — they record real failure modes where a wrong value caused
a real problem.
"""

from __future__ import annotations


# ── SERVO: gesture bookends ─────────────────────────────────────────────
HOLD_MS              = 240   # dwell after DOWN before any MOVE (long-press grab).
                             # Block Blast needs ~250 ms of stationary touch to
                             # register the piece pickup.
PRE_LIFT_MS          = 260   # settle before UP so the game commits the place.
INITIAL_LIFT_PX      = 80    # initial upward nudge so the piece pops above the
                             # finger (helps Block Blast confirm the piece is
                             # being dragged, not held).
GRAB_Y_NUDGE_PX      = 100   # queue slot icon sits above slot_center; press
                             # this far above the center to actually grab.

# ── SERVO: loop pacing ──────────────────────────────────────────────────
MAX_LOOP_S           = 2.0   # total servo budget per placement.
SETTLE_MS            = 50    # sleep after each move() so the next frame
                             # samples a settled piece.
FRAME_TIMEOUT_S      = 0.03  # how long to wait for a fresh frame per iter.
MAX_NO_PIECE_FRAMES  = 8     # consecutive frames without a detected piece
                             # before giving up.

# ── SERVO: PD controller ────────────────────────────────────────────────
GAIN                 = 0.7   # P term.  Rough estimate of piece-px per
                             # finger-px; smaller = larger steps for the
                             # same error, faster but more overshoot-prone.
DERIV_GAIN           = 1.8   # D term.  Damps overshoot by anticipating
                             # the piece's motion: when the error is
                             # shrinking (piece is already heading toward
                             # the target), the next step is reduced by
                             # roughly this multiple of the error
                             # derivative.  0.0 = pure P; 1.0 = strong
                             # damping; >2 will under-shoot and crawl.

# ── SERVO: motion speed (primary knobs) ─────────────────────────────────
# MAX_STEP_PX is the dominant speed control.  Lower = slower, smoother
# drag (and more iters to reach a far target).  The finger never travels
# more than this many frame pixels per servo iteration regardless of how
# big the error is, so big errors get walked in over multiple iters
# rather than jumping at full velocity into Block Blast's drag follower
# (which lags fast moves and gives us stale visual feedback the next
# frame).
MAX_STEP_PX          = 30

# Each iteration's step is interpolated into MOVE_SUBSTEPS touch-MOVE
# events spaced MOVE_SUBSTEP_MS apart, so Android sees a smooth drag
# instead of a single ~MAX_STEP_PX teleport.  Block Blast renders its
# drag follower much more reliably on continuous motion.
MOVE_SUBSTEPS        = 4
MOVE_SUBSTEP_MS      = 8

# ── SERVO: lock criteria ────────────────────────────────────────────────
LOCK_TOL_PX          = 16    # |err| px tolerance on both axes.
LOCK_SCORE_MIN       = 0.60  # required template-match score to release.

# ── SERVO: detection (frame-diff + template match) ──────────────────────
MATCH_SCORE_MIN      = 0.2  # below this, treat the frame as "no piece";
                             # increment the no_piece counter.
DIFF_THRESHOLD       = 25    # per-pixel grayscale abs-diff threshold for
                             # "this pixel moved since the baseline".
MORPH_KERNEL_PX      = 7     # closing kernel — fills small holes inside the
                             # piece body where the rendered colour happens
                             # to match the baseline, so the template
                             # correlates against a solid blob.


# ── AUTOPLAY: assist GUI (app_autoplay.py) ──────────────────────────────
AUTO_CONF_THRESHOLD  = 0.3   # min CNN confidence across all 3 queue slots
                             # before the assist GUI will dispatch a servo.
AUTO_POST_PLACE_MS   = 200   # cooldown after the servo completes.
AUTO_SERVO_BUDGET_MS = 2800  # outer cap on a single servo run, in ms.

# ── AUTOPLAY: headless loop (control/auto_player.py) ────────────────────
CONF_THRESHOLD       = 0.65  # skip a frame if any slot confidence is
                             # below this.
POST_PLACE_MS        = 1000   # wait after each swipe (animation + queue
                             # refresh).
CHANGE_TIMEOUT_MS    = 600   # give up waiting for a frame change after
                             # this.
DISPLAY_SCALE        = 0.45  # pygame preview window scale factor.
