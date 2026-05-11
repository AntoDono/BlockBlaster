"""Training statistics logger."""

from __future__ import annotations

from collections import deque

import param


class RunningStats:
    """Maintain a rolling window of scalar values."""

    def __init__(self, window: int = 100) -> None:
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, value: float) -> None:
        self._buf.append(value)

    @property
    def mean(self) -> float:
        return sum(self._buf) / len(self._buf) if self._buf else 0.0

    @property
    def count(self) -> int:
        return len(self._buf)


def log_epoch(
    epoch: int,
    train_loss: float,
    test_loss: float | None,
    best_test_loss: float,
) -> None:
    if epoch % param.LOG_INTERVAL == 0 or test_loss is not None:
        parts = [f"Epoch {epoch:>4d}", f"train_loss={train_loss:.4f}"]
        if test_loss is not None:
            parts.append(f"test_loss={test_loss:.4f}")
            parts.append(f"best={best_test_loss:.4f}")
        print("  ".join(parts))
