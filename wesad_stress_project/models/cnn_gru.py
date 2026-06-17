"""Hybrid CNN-GRU classifier for WESAD sequence windows."""

from __future__ import annotations

import torch
from torch import nn


class CNNGRUClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 6,
        conv_channels: int = 32,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, conv_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.gru = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])

