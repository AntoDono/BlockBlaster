import asyncio
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from get_frame import get_frame, open_device_stream  # noqa: E402


TARGET_FPS = 15
WINDOW_NAME = "iOS stream"

# A pixel "changed" if its grayscale delta exceeds this (0..255).
PIXEL_DIFF_THRESHOLD = 25
# Fraction of changed pixels that counts as a "big" change.
CHANGE_FRACTION = 0.20


def changed_fraction(prev: np.ndarray, curr: np.ndarray) -> float:
    """Fraction of pixels whose grayscale value changed by > PIXEL_DIFF_THRESHOLD."""
    if prev.shape != curr.shape:
        return 1.0
    g_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    g_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g_prev, g_curr)
    changed = np.count_nonzero(diff > PIXEL_DIFF_THRESHOLD)
    return changed / diff.size


async def stream():
    frame_dt = 1.0 / TARGET_FPS

    async with open_device_stream() as screenshot:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        prev_frame: np.ndarray | None = None
        last_log = time.monotonic()
        frames = 0
        next_deadline = time.monotonic()

        while True:
            t0 = time.monotonic()
            frame = await get_frame(screenshot)

            if frame is not None:
                if prev_frame is not None:
                    frac = changed_fraction(prev_frame, frame)
                    if frac >= CHANGE_FRACTION:
                        print(f"change: {frac * 100:.1f}% pixels differ")
                prev_frame = frame
                cv2.imshow(WINDOW_NAME, frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            frames += 1
            now = time.monotonic()
            if now - last_log >= 1.0:
                print(f"{frames} fps  (last frame {(now - t0) * 1000:.1f} ms)")
                frames = 0
                last_log = now

            next_deadline += frame_dt
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                next_deadline = time.monotonic()

        cv2.destroyAllWindows()


asyncio.run(stream())
