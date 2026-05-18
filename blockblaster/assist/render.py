"""Public re-export surface for assist GUI rendering.

All draw functions live in :mod:`render_phone` and :mod:`render_recon`.
This module re-exports everything so that existing imports like::

    from blockblaster.assist.render import draw_phone_panel, draw_recon_panel
    from blockblaster.assist.render import SUGGEST_FILL

continue to work without modification.
"""

from blockblaster.assist.render_phone import (  # noqa: F401
    DEVICE_ERR_COL,
    DEVICE_OK_COL,
    DIM_TEXT,
    DRAG_COLOR,
    LABEL_COL,
    OVERLAY_BOX,
    OVERLAY_DOT,
    OVERLAY_GRID,
    PANEL_BG,
    PANEL_BORDER,
    QUEUE_BOX_COLOR,
    QUEUE_DIVIDER,
    STATUS_BG,
    SUGGEST_BORDER,
    SUGGEST_FILL,
    SUGGEST_FILL_A,
    TEXT_COLOR,
    bgr_to_surface,
    draw_calib_target_on_phone,
    draw_drag_preview,
    draw_grid_overlay,
    draw_phone_panel,
    draw_queue_overlay,
    draw_status_bar,
    draw_suggestion_on_phone,
    draw_swipe_arrow_on_phone,
)
from blockblaster.assist.render_recon import draw_recon_panel  # noqa: F401
