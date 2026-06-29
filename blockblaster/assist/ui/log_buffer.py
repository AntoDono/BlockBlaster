"""Ring buffer of assist log lines for the GUI log panel."""

from __future__ import annotations

from collections import deque
from typing import Deque

LOG_MAX_LINES = 1000


def append_log(lines: Deque[str], msg: str) -> None:
    print(msg)
    lines.append(msg)
