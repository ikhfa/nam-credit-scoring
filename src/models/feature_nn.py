"""Single-feature sub-network for Neural Additive Models."""

import torch
import torch.nn as nn


class FeatureNN(nn.Module):
    """Small MLP that processes a single input feature and outputs a scalar contribution.

    Architecture: Input(1) -> [Linear -> ReLU -> Dropout] x N -> Linear(1)
    """

    def __init__(
        self,
        hidden_sizes: list[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        layers = []
        in_size = 1
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, 1) — single feature values.

        Returns:
            Tensor of shape (batch_size, 1) — feature contribution.
        """
        return self.net(x)
