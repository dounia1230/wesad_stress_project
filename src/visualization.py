"""Safe activation capture utilities for trained convolutional models."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def extract_feature_maps(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    layers: Sequence[torch.nn.Module],
) -> list[torch.Tensor]:
    """Capture detached CPU outputs from selected layers during one inference pass."""
    if not layers:
        raise ValueError("At least one layer must be supplied.")
    activations: list[torch.Tensor] = []
    handles = []

    def capture(_module, _arguments, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("The hooked layer did not return a tensor.")
        activations.append(output.detach().cpu())

    try:
        handles = [layer.register_forward_hook(capture) for layer in layers]
        model.eval()
        parameter = next(model.parameters())
        with torch.no_grad():
            model(inputs.to(parameter.device))
    finally:
        for handle in handles:
            handle.remove()
    if len(activations) != len(layers):
        raise RuntimeError("Not all requested hooks were invoked.")
    return activations
