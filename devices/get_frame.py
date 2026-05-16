from __future__ import annotations

from contextlib import asynccontextmanager

import cv2
import numpy as np
from pymobiledevice3.services.dvt.instruments.dvt_provider import DvtProvider
from pymobiledevice3.services.dvt.instruments.screenshot import Screenshot
from pymobiledevice3.tunneld.api import get_tunneld_devices


async def get_frame(screenshot: Screenshot) -> np.ndarray | None:
    """Grab one screenshot from an already-open Screenshot service and decode to BGR.

    Returns None if the PNG failed to decode.
    """
    png = await screenshot.get_screenshot()
    arr = np.frombuffer(png, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@asynccontextmanager
async def open_device_stream():
    """Open tunneld -> DVT -> Screenshot in one shot.

    Usage:
        async with open_device_stream() as screenshot:
            frame = await get_frame(screenshot)
    """
    rsds = await get_tunneld_devices()
    if not rsds:
        raise RuntimeError("No devices via tunneld. Is tunneld running?")
    rsd = rsds[0]
    try:
        async with DvtProvider(rsd) as dvt, Screenshot(dvt) as screenshot:
            yield screenshot
    finally:
        await rsd.close()
