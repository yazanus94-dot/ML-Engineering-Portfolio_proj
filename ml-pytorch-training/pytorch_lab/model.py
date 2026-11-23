"""Small MLP with a documented checkpoint format."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, hidden: tuple[int, ...] = (32, 32), num_classes: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model(config: dict) -> MLP:
    """Recreate a model from a checkpoint configuration dictionary."""
    return MLP(
        input_dim=int(config["input_dim"]),
        hidden=tuple(config["hidden"]),
        num_classes=int(config["num_classes"]),
    )


def load_checkpoint(path: str):
    """Load ``(model, checkpoint)``; validates the checkpoint schema."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("config", "state_dict", "metrics", "version"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing required key: {key}")
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
