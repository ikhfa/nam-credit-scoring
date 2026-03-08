"""Tests for NAM model architecture and forward pass."""

import pytest
import numpy as np
import torch

from src.models.feature_nn import FeatureNN
from src.models.nam import NAM


class TestFeatureNN:
    def test_output_shape(self):
        net = FeatureNN(hidden_sizes=[32, 32], dropout=0.1)
        x = torch.randn(16, 1)
        out = net(x)
        assert out.shape == (16, 1)

    def test_single_input(self):
        net = FeatureNN()
        x = torch.tensor([[0.5]])
        out = net(x)
        assert out.shape == (1, 1)

    def test_deterministic_eval(self):
        net = FeatureNN(hidden_sizes=[32, 32], dropout=0.5)
        net.eval()
        x = torch.randn(10, 1)
        out1 = net(x)
        out2 = net(x)
        assert torch.allclose(out1, out2)


class TestNAM:
    @pytest.fixture
    def model(self):
        return NAM(num_features=5, hidden_sizes=[16, 16], dropout=0.1)

    def test_output_shapes(self, model):
        x = torch.randn(32, 5)
        logit, contributions = model(x)

        assert logit.shape == (32, 1)
        assert len(contributions) == 5
        for c in contributions:
            assert c.shape == (32, 1)

    def test_additivity(self, model):
        """Verify that logit = sum(contributions) + bias."""
        model.eval()
        x = torch.randn(8, 5)
        logit, contributions = model(x)

        manual_sum = sum(c for c in contributions)  # (8, 1)
        manual_logit = manual_sum + model.bias

        assert torch.allclose(logit, manual_logit, atol=1e-5)

    def test_shape_function(self, model):
        model.eval()
        values = torch.linspace(-3, 3, 50)
        output = model.get_shape_function(0, values)
        assert output.shape == (50,)

    def test_get_all_shape_functions(self, model):
        model.eval()
        ranges = {i: torch.linspace(-2, 2, 20) for i in range(5)}
        results = model.get_all_shape_functions(ranges)

        assert len(results) == 5
        for idx, (vals, outs) in results.items():
            assert vals.shape == (20,)
            assert outs.shape == (20,)

    def test_feature_dropout(self):
        model = NAM(num_features=3, hidden_sizes=[8], feature_dropout=1.0)
        model.train()
        x = torch.randn(4, 3)
        logit, contributions = model(x)

        # With dropout=1.0, all contributions should be zero
        for c in contributions:
            assert torch.allclose(c, torch.zeros_like(c))

    def test_num_parameters(self):
        model = NAM(num_features=23, hidden_sizes=[64, 64, 64], dropout=0.3)
        n_params = sum(p.numel() for p in model.parameters())
        # Each sub-net: (1*64+64) + (64*64+64) + (64*64+64) + (64*1+1) = 8513
        # 23 sub-nets + 1 bias = 23*8513 + 1 = 195800
        assert n_params > 100000  # sanity check
        print(f"Total parameters: {n_params:,}")
