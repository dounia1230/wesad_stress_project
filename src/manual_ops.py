"""Educational implementations of elementary two-dimensional CNN operations."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def corr2d(
    input_matrix: torch.Tensor,
    kernel: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """Cross-correlate two 2D tensors and return a 2D output matrix."""
    if input_matrix.ndim != 2 or kernel.ndim != 2:
        raise ValueError("input_matrix and kernel must both be two-dimensional.")
    if stride <= 0 or padding < 0:
        raise ValueError("stride must be positive and padding must be non-negative.")
    padded = F.pad(input_matrix, (padding, padding, padding, padding))
    kh, kw = kernel.shape
    height = (padded.shape[0] - kh) // stride + 1
    width = (padded.shape[1] - kw) // stride + 1
    if height <= 0 or width <= 0:
        raise ValueError("The kernel is larger than the padded input.")
    output = input_matrix.new_empty((height, width))
    for row in range(height):
        for col in range(width):
            patch = padded[row * stride : row * stride + kh, col * stride : col * stride + kw]
            output[row, col] = torch.sum(patch * kernel)
    return output


def manual_max_pool2d(
    input_matrix: torch.Tensor,
    kernel_size: int,
    stride: int | None = None,
) -> torch.Tensor:
    """Max-pool one 2D matrix without using a PyTorch pooling operation."""
    return _manual_pool2d(input_matrix, kernel_size, stride, reduction="max")


def manual_avg_pool2d(
    input_matrix: torch.Tensor,
    kernel_size: int,
    stride: int | None = None,
) -> torch.Tensor:
    """Average-pool one 2D matrix without using a PyTorch pooling operation."""
    return _manual_pool2d(input_matrix, kernel_size, stride, reduction="mean")


def _manual_pool2d(
    input_matrix: torch.Tensor,
    kernel_size: int,
    stride: int | None,
    reduction: str,
) -> torch.Tensor:
    if input_matrix.ndim != 2:
        raise ValueError("input_matrix must be two-dimensional.")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive.")
    stride = kernel_size if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive.")
    height = (input_matrix.shape[0] - kernel_size) // stride + 1
    width = (input_matrix.shape[1] - kernel_size) // stride + 1
    if height <= 0 or width <= 0:
        raise ValueError("kernel_size is larger than the input.")
    output = input_matrix.new_empty((height, width))
    for row in range(height):
        for col in range(width):
            patch = input_matrix[
                row * stride : row * stride + kernel_size,
                col * stride : col * stride + kernel_size,
            ]
            output[row, col] = patch.max() if reduction == "max" else patch.mean()
    return output
