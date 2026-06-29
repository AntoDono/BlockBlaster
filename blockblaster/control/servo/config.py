"""Tunable constants for the visual servo loop."""

# Grab / gesture timing
HOLD_MS = 240
PRE_LIFT_MS = 260
INITIAL_LIFT_PX = 60
MIN_INITIAL_LIFT_PX = 30
INITIAL_LIFT_SETTLE_MS = 150
INITIAL_LIFT_SUBSTEPS = 8
INITIAL_LIFT_SUBSTEP_MS = 12
START_NOISE_X_PX = 30

# Loop pacing
MAX_LOOP_S = 7.0
SETTLE_MS = 50
FRAME_TIMEOUT_S = 0.03
MAX_NO_PIECE_FRAMES = 60

# PD controller
GAIN = 0.7
DERIV_GAIN = 1.8
MAX_STEP_PX = 50
FINE_STEP_PX = 10
MOVE_SUBSTEPS = 4
MOVE_SUBSTEP_MS = 8

# Approach / lock
APPROACH_RADIUS_PX = 300
ROI_MARGIN_PX = APPROACH_RADIUS_PX // 2
RECON_LOCK_FRAMES = 4
BOUNDARY_TOL_PX = 30
MATCH_SCORE_MIN = 0.20

# Detection
DIFF_THRESHOLD = 25
MORPH_KERNEL_PX = 7
MIN_MOVED_PX = 60
PIECE_AREA_FRAC = 0.22

# Motion-blob area guard (sampled after lift)
INITIAL_AREA_DELAY_S = 0.5
AREA_GROW_RELEASE_RATIO = 1.80   # row-clear / completion glow merged with piece
AREA_SHRINK_PUSH_RATIO = 0.60   # piece slipping off board → push toward center
AREA_GROW_STREAK = 2
AREA_SHRINK_STREAK = 2
AREA_GROW_HOLD_STEP_PX = 5   # max movement per axis while row-clear glow is active
