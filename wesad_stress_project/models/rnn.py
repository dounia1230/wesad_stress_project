"""Simple recurrent baseline for WESAD sequence windows."""

from __future__ import annotations

import torch
from torch import nn


class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        final_hidden = hidden[-1]
        return self.classifier(final_hidden)

