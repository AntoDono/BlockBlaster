"""Central configuration for BlockBlaster auto-play.

Auto-play loop tunables live in :mod:`.params` and are re-exported here
so callers can write ``from blockblaster.config import CONF_THRESHOLD``.
Servo-internal tunables now live at the top of
:mod:`blockblaster.control.servo`.
"""

from blockblaster.config.params import *  # noqa: F401,F403
