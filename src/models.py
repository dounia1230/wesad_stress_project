"""Neural network architectures used by the WESAD notebooks."""

from __future__ import annotations

import torch
from torch import nn


def initialize_linear_layers(
    model: nn.Module,
    method: str,
    gaussian_std: float = 0.02,
    constant_value: float = 0.01,
) -> None:
    """Initialize every Linear layer using Gaussian, constant, or Xavier weights.

    Batch-normalization parameters retain their PyTorch defaults. Linear biases
    are set to zero for all methods so validation compares only weight schemes.
    """
    if method not in {"gaussian", "constant", "xavier"}:
        raise ValueError("method must be 'gaussian', 'constant', or 'xavier'.")
    if gaussian_std <= 0:
        raise ValueError("gaussian_std must be positive.")
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if method == "gaussian":
                nn.init.normal_(module.weight, mean=0.0, std=gaussian_std)
            elif method == "constant":
                nn.init.constant_(module.weight, constant_value)
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WESADScalogramCNN(nn.Module):
    """2D CNN for inputs shaped ``(batch, 3, 64, 64)``.

    The forward method returns one raw logit per sample; sigmoid is intentionally
    left to evaluation code so the model is compatible with BCEWithLogitsLoss.
    """

    def __init__(
        self,
        input_channels: int = 3,
        filters: tuple[int, int, int] = (16, 32, 64),
        padding: int = 1,
        conv_stride: int = 1,
        pooling_type: str = "max",
        use_pointwise_conv: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(filters) != 3 or any(value <= 0 for value in filters):
            raise ValueError("filters must contain three positive integers.")
        if padding not in {0, 1}:
            raise ValueError("padding must be 0 or 1 for the planned ablation.")
        if conv_stride not in {1, 2}:
            raise ValueError("conv_stride must be 1 or 2.")
        pools = {"max": nn.MaxPool2d, "avg": nn.AvgPool2d}
        if pooling_type not in pools:
            raise ValueError("pooling_type must be 'max' or 'avg'.")
        pool = pools[pooling_type]
        f1, f2, f3 = filters
        # Stride 2, when selected, is applied only to the first convolution.
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, f1, 3, stride=conv_stride, padding=padding),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            pool(2),
            nn.Conv2d(f1, f2, 3, padding=padding),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            pool(2),
            nn.Conv2d(f2, f3, 3, padding=padding),
            nn.BatchNorm2d(f3),
            nn.ReLU(),
        )
        output_channels = 32 if use_pointwise_conv else f3
        self.pointwise = (
            nn.Sequential(nn.Conv2d(f3, output_channels, 1), nn.ReLU())
            if use_pointwise_conv
            else nn.Identity()
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(output_channels * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected (batch, 3, H, W), received {tuple(x.shape)}.")
        return self.classifier(self.pointwise(self.features(x)))


class SimpleRNNClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 32,
        dropout: float = 0.3,
    ) -> None:
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


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
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
        _, (hidden, _) = self.lstm(x)
        final_hidden = hidden[-1]
        return self.classifier(final_hidden)
