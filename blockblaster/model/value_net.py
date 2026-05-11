"""CNN value network v(s) -> scalar."""

from __future__ import annotations

import torch
import torch.nn as nn

import param


class ValueNet(nn.Module):
    """
    Small CNN that maps a (1 + QUEUE_SIZE, 8, 8) state tensor to a scalar value.

    Architecture:
        Conv2d(in_channels, C, 3, padding=1) -> ReLU
        Conv2d(C, C, 3, padding=1)           -> ReLU
        Conv2d(C, C, 3, padding=1)           -> ReLU
        Flatten -> Linear(C * 64, H) -> ReLU -> Linear(H, 1)
    """

    def __init__(
        self,
        in_channels: int | None = None,
        cnn_channels: int | None = None,
        hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        in_ch = in_channels if in_channels is not None else 1 + param.QUEUE_SIZE
        C = cnn_channels if cnn_channels is not None else param.CNN_CHANNELS
        H = hidden_size if hidden_size is not None else param.HIDDEN_SIZE

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C * param.BOARD_SIZE * param.BOARD_SIZE, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, 8, 8) -> (N, 1)."""
        return self.head(self.conv(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: eval mode, no grad, returns (N,) tensor."""
        training = self.training
        self.eval()
        out = self(x).squeeze(1)
        self.train(training)
        return out
