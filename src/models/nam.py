"""Neural Additive Model for binary classification."""

import torch
import torch.nn as nn

from src.models.feature_nn import FeatureNN


class NAM(nn.Module):
    """Neural Additive Model.

    Prediction: logit = bias + sum_i f_i(x_i)
    where each f_i is a FeatureNN sub-network.

    The model is inherently interpretable because each feature's contribution
    can be visualized as a shape function f_i(x_i).
    """

    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int] = None,
        dropout: float = 0.3,
        feature_dropout: float = 0.0,
    ):
        """
        Args:
            num_features: Number of input features (each gets its own sub-network).
            hidden_sizes: Hidden layer sizes for each FeatureNN.
            dropout: Dropout rate within each sub-network.
            feature_dropout: Probability of zeroing out an entire feature sub-network
                during training (regularization).
        """
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64, 64]

        self.num_features = num_features
        self.feature_dropout = feature_dropout

        self.feature_nns = nn.ModuleList([
            FeatureNN(hidden_sizes=hidden_sizes, dropout=dropout)
            for _ in range(num_features)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, num_features).

        Returns:
            logit: Tensor of shape (batch_size, 1) — raw logit (pre-sigmoid).
            feature_contributions: List of num_features tensors, each (batch_size, 1).
        """
        feature_contributions = []
        for i in range(self.num_features):
            fi = self.feature_nns[i](x[:, i : i + 1])  # (B, 1)

            # Feature dropout: zero out entire sub-network output during training
            if self.training and self.feature_dropout > 0:
                mask = torch.bernoulli(
                    torch.full_like(fi, 1.0 - self.feature_dropout)
                )
                fi = fi * mask

            feature_contributions.append(fi)

        # Sum all contributions + bias
        stacked = torch.cat(feature_contributions, dim=1)  # (B, F)
        logit = stacked.sum(dim=1, keepdim=True) + self.bias  # (B, 1)

        return logit, feature_contributions

    def get_shape_function(
        self, feature_idx: int, values: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate shape function f_i for a range of values.

        Args:
            feature_idx: Index of the feature.
            values: 1D tensor of input values to evaluate.

        Returns:
            1D tensor of shape function outputs f_i(values).
        """
        self.eval()
        with torch.no_grad():
            x = values.unsqueeze(1)  # (N, 1)
            output = self.feature_nns[feature_idx](x)  # (N, 1)
        return output.squeeze(1)

    def get_all_shape_functions(
        self, feature_ranges: dict[int, torch.Tensor]
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Extract shape functions for multiple features.

        Args:
            feature_ranges: Dict mapping feature index to 1D tensor of values.

        Returns:
            Dict mapping feature index to (values, f_i(values)) tuples.
        """
        results = {}
        for idx, values in feature_ranges.items():
            outputs = self.get_shape_function(idx, values)
            results[idx] = (values, outputs)
        return results
