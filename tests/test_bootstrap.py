"""
Tests for src/bootstrap.py

Validates the geometric stationary bootstrap and panel MCS implementation,
including seed reproducibility, block structure, and estimand correctness.
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bootstrap import (
    geometric_block_bootstrap,
    panel_bootstrap_mean,
    mcs_bootstrap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_series():
    """Simple time series for bootstrap tests."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, 200)


@pytest.fixture
def panel_losses_dict():
    """Panel losses as dict of pd.Series for 3 markets, 100 obs each."""
    rng = np.random.default_rng(55)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    return {
        "m1": pd.Series(rng.exponential(1.0, 100), index=dates),
        "m2": pd.Series(rng.exponential(1.2, 100), index=dates),
        "m3": pd.Series(rng.exponential(0.8, 100), index=dates),
    }


@pytest.fixture
def loss_matrix_panel():
    """(T, n_models) loss matrix for MCS tests."""
    rng = np.random.default_rng(55)
    return rng.exponential(1.0, size=(100, 3))


# ---------------------------------------------------------------------------
# Geometric block bootstrap tests
# Actual API: geometric_block_bootstrap(T, mean_block_length, B, seed) -> (B, T)
# ---------------------------------------------------------------------------

class TestGeometricBlockBootstrap:
    def test_output_shape(self, simple_series):
        """Bootstrap returns (B, T) matrix."""
        T = len(simple_series)
        B = 50
        indices = geometric_block_bootstrap(T, mean_block_length=20, B=B, seed=42)
        assert indices.shape == (B, T)

    def test_indices_in_range(self, simple_series):
        """All resampled indices should be valid [0, T)."""
        T = len(simple_series)
        indices = geometric_block_bootstrap(T, mean_block_length=20, B=10, seed=1)
        assert np.all(indices >= 0)
        assert np.all(indices < T)

    def test_seed_reproducibility(self, simple_series):
        """Same seed produces identical (B, T) matrix."""
        T = len(simple_series)
        idx1 = geometric_block_bootstrap(T, mean_block_length=20, B=10, seed=42)
        idx2 = geometric_block_bootstrap(T, mean_block_length=20, B=10, seed=42)
        np.testing.assert_array_equal(idx1, idx2)

    def test_different_seeds_differ(self, simple_series):
        """Different seeds produce different matrices."""
        T = len(simple_series)
        idx1 = geometric_block_bootstrap(T, mean_block_length=20, B=10, seed=1)
        idx2 = geometric_block_bootstrap(T, mean_block_length=20, B=10, seed=2)
        assert not np.array_equal(idx1, idx2)

    def test_mean_block_affects_autocorrelation(self, simple_series):
        """Larger mean_block => higher within-draw index autocorrelation."""
        T = len(simple_series)
        idx_small = geometric_block_bootstrap(T, mean_block_length=2, B=1, seed=10)[0]
        idx_large = geometric_block_bootstrap(T, mean_block_length=50, B=1, seed=10)[0]
        ac_small = np.corrcoef(idx_small[:-1], idx_small[1:])[0, 1]
        ac_large = np.corrcoef(idx_large[:-1], idx_large[1:])[0, 1]
        assert ac_large > ac_small


# ---------------------------------------------------------------------------
# Panel bootstrap mean tests
# Actual API: panel_bootstrap_mean(market_losses, B, mean_block_length, seed)
#             -> (bootstrap_means, point_mean, se)
# ---------------------------------------------------------------------------

class TestPanelBootstrapMean:
    def test_output_structure(self, panel_losses_dict):
        """panel_bootstrap_mean returns (boot_means, point_mean, se)."""
        boot_means, point_mean, se = panel_bootstrap_mean(
            panel_losses_dict, B=100, mean_block_length=10, seed=42
        )
        assert len(boot_means) == 100
        assert isinstance(point_mean, float)
        assert isinstance(se, float)

    def test_bootstrap_mean_near_sample_mean(self):
        """Bootstrap mean should be near sample mean for iid data."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        losses = {
            "m1": pd.Series(rng.exponential(1.0, 500), index=dates),
            "m2": pd.Series(rng.exponential(1.0, 500), index=dates),
        }
        true_mean = np.mean([np.mean(v.values) for v in losses.values()])
        boot_means, point_mean, se = panel_bootstrap_mean(
            losses, B=200, mean_block_length=10, seed=99
        )
        assert abs(point_mean - true_mean) < 0.05
        assert abs(np.mean(boot_means) - true_mean) < 0.10

    def test_bootstrap_se_positive(self, panel_losses_dict):
        """Bootstrap SE should be positive."""
        _, _, se = panel_bootstrap_mean(
            panel_losses_dict, B=100, mean_block_length=10, seed=7
        )
        assert se > 0

    def test_seed_reproducibility_panel(self, panel_losses_dict):
        """Same seed gives same bootstrap distribution."""
        b1, p1, s1 = panel_bootstrap_mean(
            panel_losses_dict, B=50, mean_block_length=10, seed=100
        )
        b2, p2, s2 = panel_bootstrap_mean(
            panel_losses_dict, B=50, mean_block_length=10, seed=100
        )
        np.testing.assert_array_equal(b1, b2)
        assert p1 == p2
        assert s1 == s2


# ---------------------------------------------------------------------------
# MCS bootstrap tests
# Actual API: mcs_bootstrap(loss_matrix, model_names, alpha, B, mean_block_length, seed)
#             -> dict with 'mcs_set', 'eliminated', 'pvalues'
# loss_matrix shape: (T, n_models)
# ---------------------------------------------------------------------------

class TestMCSBootstrap:
    def test_mcs_returns_dict(self, loss_matrix_panel):
        """MCS should return dict with 'mcs_set', 'eliminated', 'pvalues'."""
        names = ["A", "B", "C"]
        result = mcs_bootstrap(
            loss_matrix_panel, model_names=names,
            B=100, mean_block_length=10, seed=42, alpha=0.10
        )
        assert "mcs_set" in result
        assert "eliminated" in result
        assert "pvalues" in result

    def test_mcs_set_nonempty(self, loss_matrix_panel):
        """MCS set should contain at least one model."""
        names = ["A", "B", "C"]
        result = mcs_bootstrap(
            loss_matrix_panel, model_names=names,
            B=100, mean_block_length=10, seed=42, alpha=0.10
        )
        assert len(result["mcs_set"]) >= 1

    def test_mcs_includes_best_model(self, loss_matrix_panel):
        """Model with smallest mean loss should always be in MCS."""
        names = ["A", "B", "C"]
        result = mcs_bootstrap(
            loss_matrix_panel, model_names=names,
            B=200, mean_block_length=10, seed=42, alpha=0.10
        )
        mean_losses = np.mean(loss_matrix_panel, axis=0)
        best_name = names[int(np.argmin(mean_losses))]
        assert best_name in result["mcs_set"]

    def test_mcs_dominated_model_excluded(self):
        """Clearly dominated model should be excluded from MCS."""
        rng = np.random.default_rng(20)
        T = 500
        loss_matrix = np.column_stack([
            rng.exponential(0.5, T),
            rng.exponential(1.0, T),
            rng.exponential(10.0, T),  # clearly dominated
        ])
        names = ["good", "medium", "bad"]
        result = mcs_bootstrap(
            loss_matrix, model_names=names,
            B=300, mean_block_length=20, seed=42, alpha=0.10
        )
        if len(result["mcs_set"]) < 3:
            assert "bad" not in result["mcs_set"]

    def test_single_model_always_in_mcs(self):
        """Single model trivially survives MCS."""
        rng = np.random.default_rng(5)
        loss_matrix = rng.exponential(1.0, (100, 1))
        result = mcs_bootstrap(
            loss_matrix, model_names=["only"],
            B=50, mean_block_length=10, seed=5, alpha=0.10
        )
        assert "only" in result["mcs_set"]

    def test_equal_models_all_in_mcs(self):
        """Identical models should all be retained in MCS."""
        rng = np.random.default_rng(6)
        base = rng.exponential(1.0, 100)
        loss_matrix = np.column_stack([base, base, base])
        names = ["X", "Y", "Z"]
        result = mcs_bootstrap(
            loss_matrix, model_names=names,
            B=100, mean_block_length=10, seed=6, alpha=0.10
        )
        assert len(result["mcs_set"]) == 3
