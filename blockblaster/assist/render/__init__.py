"""Public re-export surface for assist GUI rendering."""

from blockblaster.assist.render.phone import (  # noqa: F401
    DEVICE_ERR_COL,
    DEVICE_OK_COL,
    DIM_TEXT,
    LABEL_COL,
    PANEL_BG,
    PANEL_BORDER,
    STATUS_BG,
    SUGGEST_BORDER,
    SUGGEST_FILL,
    SUGGEST_FILL_A,
    TEXT_COLOR,
    bgr_to_surface,
    draw_phone_panel,
    draw_status_bar,
)
from blockblaster.assist.render.recon import draw_recon_panel  # noqa: F401
from blockblaster.assist.render.cnn_debug import draw_cnn_debug_panel  # noqa: F401
from blockblaster.assist.render.logs import draw_log_panel  # noqa: F401
from blockblaster.assist.render.frame_diff import draw_frame_diff_panel  # noqa: F401
