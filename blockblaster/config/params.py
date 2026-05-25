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
INITIAL_LIFT_PX      = 300    # initial upward nudge so the piece pops above the
                             # finger (helps Block Blast confirm the piece is
                             # being dragged, not held).
GRAB_Y_NUDGE_PX      = 100   # queue slot icon sits above slot_center; press
                             # this far above the center to actually grab.
PRELIFT_CONFIRM_S    = 0.25   # after the pre-lift to board centre, wait up
                             # to this long for the matcher to confirm the
                             # piece is rendered before activating the PD
                             # loop.  Avoids the PD steering off half-
                             # drawn frames while Block Blast's drag
                             # follower catches up.  Abort if exceeded.

# ── SERVO: loop pacing ──────────────────────────────────────────────────
MAX_LOOP_S           = 3.0   # total servo budget per placement.
SETTLE_MS            = 50    # sleep after each move() so the next frame
                             # samples a settled piece.
FRAME_TIMEOUT_S      = 0.05  # how long to wait for a fresh frame per iter.
MAX_NO_PIECE_FRAMES  = 16    # consecutive frames without a detected piece
                             # before giving up.  Higher = more patience
                             # when the matcher briefly loses the piece
                             # (occlusion, animation flash, etc.).

# ── SERVO: PD controller ────────────────────────────────────────────────
GAIN                 = 1.5   # P term.  Rough estimate of piece-px per
                             # finger-px; smaller = larger steps for the
                             # same error, faster but more overshoot-prone.
DERIV_GAIN           = 2.5   # D term.  Damps overshoot by anticipating
                             # the piece's motion: when the error is
                             # shrinking (piece is already heading toward
                             # the target), the next step is reduced by
                             # roughly this multiple of the error
                             # derivative.  0.0 = pure P; 1.0 = strong
                             # damping; >2 will under-shoot and crawl.

# ── SERVO: motion speed (primary knobs) ─────────────────────────────────
# Step ceiling is distance-adaptive: big jumps when far from target,
# small precise steps when close.  The PD controller's P term already
# scales with error, but we still need an upper bound per iter so we
# don't outrun Block Blast's drag follower (it lags fast moves and
# gives us stale visual feedback on the next frame).
#
#   |err| >= FAR_ERR  → clamp to STEP_FAR   (cover ground)
#   |err| <= NEAR_ERR → clamp to STEP_NEAR  (fine alignment)
#   in between        → linearly interpolated
#
# Expressed as multiples of one board cell so the servo just-works
# across different phone resolutions / calibration boxes without
# re-tuning.  Resolved to pixels inside place() once cfg.grid is known.
STEP_FAR_CELLS       = 1.0   # ceiling when |err| >= FAR_ERR (≈ one cell/iter).
STEP_NEAR_CELLS      = 0.2   # ceiling when |err| <= NEAR_ERR (≈ a sixth of a cell).
FAR_ERR_CELLS        = 2.5   # error magnitude considered "far" (≈ 2-3 cells).
NEAR_ERR_CELLS       = 1.0   # error magnitude considered "near" (≈ one cell).

# Each iteration's step is interpolated into MOVE_SUBSTEPS touch-MOVE
# events spaced MOVE_SUBSTEP_MS apart, so Android sees a smooth drag
# instead of a single ~one-cell teleport.  Block Blast renders its
# drag follower much more reliably on continuous motion.
MOVE_SUBSTEPS        = 64
MOVE_SUBSTEP_MS      = 2

# ── SERVO: lock criteria ────────────────────────────────────────────────
LOCK_TOL_PX          = 6    # |err| px tolerance on both axes.
LOCK_SCORE_MIN       = 0.30  # required template-match score to release.
LOCK_MIN_ANCHORS     = 5     # minimum visible anchors (out of 5) the
                             # release will accept.  Was "all 5" before,
                             # but edge placements (corner cells off the
                             # board, occluded by score popups, etc.)
                             # routinely show only 2-3 anchors even when
                             # the piece is dead on target → servo
                             # burned the whole budget at zero error.

# ── SERVO: detection (frame-diff + template match) ──────────────────────
MATCH_SCORE_MIN      = 0.5  # below this, treat the frame as "no piece";
                             # increment the no_piece counter.
DIFF_THRESHOLD       = 25    # per-pixel grayscale abs-diff threshold for
                             # "this pixel moved" — used by both the
                             # pre-DOWN baseline mask and the rolling
                             # frame-to-frame mask.
MORPH_KERNEL_CELLS   = 0.11  # closing kernel as a fraction of one cell.
                             # Fills small holes inside the piece body
                             # where the rendered colour happens to
                             # match the baseline, so the template
                             # correlates against a solid blob.
                             # Resolved to an odd px count in servo.

# ── SERVO: rolling-diff translation gate ────────────────────────────────
# The baseline diff lights up *everything* that differs from the pre-DOWN
# board — including mid-drag changes like row/column-clear glow previews,
# score popups, etc.  Once lit they stay lit (baseline never refreshes)
# and the template matcher can latch onto the wrong blob, freezing the
# reported piece position while the real piece is somewhere else.
#
# A rolling diff (current vs previous frame) is silent on steady-state
# glow — only pixels that *moved this frame* light up.  We use it as a
# *gate*, not a matcher: trust the baseline-mask detection only when the
# rolling mask has enough coverage inside the matched piece footprint.
# (Shares DIFF_THRESHOLD with the baseline diff — no reason to tune them
# separately.)
ROLLING_GATE_MIN_RATIO   = 0.05 # min fraction of the piece footprint
                                # (cells × cell_h × cell_w) that must
                                # contain rolling-mask pixels for the
                                # detection to be trusted.  Low enough
                                # that a piece whose leading edge alone
                                # moved this frame still passes; high
                                # enough that pure glow with a stationary
                                # piece fails.
ROLLING_GATE_RELAX_FACTOR = 4.0 # bypass the gate when both axes are
                                # within (this × LOCK_TOL_PX) of the
                                # target.  Near lock-in the piece has
                                # effectively stopped moving frame-to-
                                # frame; we'd otherwise lose it right
                                # at the goal.

# ── SERVO: local search window ──────────────────────────────────────────
# cv2.matchTemplate's global argmax can land on stationary debris (a
# row/column-clear glow, an already-placed cell that brightened, a score
# popup) hundreds of pixels from where the piece actually is — and
# happily report score≈1.0 because the debris happens to be piece-
# shaped.  Once seeded with a trusted location (from the pre-lift
# confirmation match), every subsequent matchTemplate is restricted to
# a window of this half-extent around `last_trusted_tl + commanded_dx,dy`.
# Phantoms outside the window are unreachable by construction.
SEARCH_RADIUS_CELLS  = 4.0   # half-extent of the matchTemplate window,
                             # in cells.  Generously contains where the
                             # piece could be after one iter even if
                             # Block Blast's drag follower lagged the
                             # commanded step, but tiny vs the full
                             # 8-cell board so faraway debris can't win.

# ── SERVO: pre-clear glow early release ─────────────────────────────────
# When the held piece is hovering over a placement that would complete a
# row/column, Block Blast renders a glow preview over the cells that
# would clear.  That glow lights up the motion-diff mask far beyond the
# piece's own footprint and confuses the template matcher (suddenly
# "the piece" looks like a whole row).  The piece is also, by definition,
# already at an optimal placement when the glow appears — so we just
# release.  Persistence guard avoids reacting to single-frame flashes
# (score popups, transient animations) that aren't actual row-clear
# previews.
GLOW_AREA_RATIO      = 1.2   # motion-mask area / piece silhouette area.
                             # Above this = much more is lit up than just
                             # the piece → likely a row-clear glow.
GLOW_HOLD_S          = 1.0   # sustained duration above the ratio before
                             # we commit.  Long enough to filter out
                             # transient flashes we didn't intend.

# ── SERVO: off-board corner recovery ────────────────────────────────────
# When a piece drifts partially off the board, the corner anchors on the
# off-side stop reporting (their cell windows have no motion to detect).
# Compare the set of *currently visible* corner anchors against the set
# of corners we'd expect to see *at the target* (corners whose cell is
# on-board at target).  If a corner that should be visible isn't,
# sustained for OFF_BOARD_HOLD_S, override the PD step with a fixed
# inward nudge until the missing corner re-appears.  Direction comes
# from which side's corners are missing (e.g. TR+BR missing → push -x).
OFF_BOARD_HOLD_S     = 0.15  # sustained duration of "missing corner that
                             # should be visible at target" before the
                             # recovery override fires.  Filters out
                             # transient occlusions (score popups, level
                             # animations, etc.) that briefly drop a
                             # corner without the piece actually being
                             # off-board.
RECOVERY_STEP_CELLS  = 0.4   # per-iter recovery step magnitude, in
                             # cells.  Slow enough to give the matcher
                             # time to re-acquire the corner once it
                             # comes back on-board, but big enough to
                             # actually unstick the piece (fractions of
                             # a cell got swallowed by the matcher's
                             # resolution on test runs).


# ── AUTOPLAY: assist GUI (app_autoplay.py) ──────────────────────────────
AUTO_CONF_THRESHOLD  = 0.4   # min CNN confidence across all 3 queue slots
                             # before the assist GUI will dispatch a servo.
AUTO_POST_PLACE_MS   = 1500   # cooldown after the servo completes.
AUTO_SERVO_BUDGET_MS = 3000  # outer cap on a single servo run, in ms.

# ── AUTOPLAY: headless loop (control/auto_player.py) ────────────────────
CONF_THRESHOLD       = 0.3  # skip a frame if any slot confidence is
                             # below this.
POST_PLACE_MS        = 300   # wait after each swipe (animation + queue
                             # refresh).
CHANGE_TIMEOUT_MS    = 600   # give up waiting for a frame change after
                             # this.
DISPLAY_SCALE        = 0.45  # pygame preview window scale factor.
