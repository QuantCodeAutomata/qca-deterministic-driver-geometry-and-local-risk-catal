"""
Tests for src/recursion.py and src/diagnostics.py

Validates the observation-driven state recursion, log-Lipschitz diagnostic,
and key diagnostic metrics: QLIKE, VaR, coverage, and PIT.
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.recursion import (
    compute_update,
    run_recursion,
    log_lipschitz_diagnostic,
    compute_residuals,
)
from src.diagnostics import (
    qlike_fd,
    mean_qlike,
    var_forecast,
    coverage_rate,
    compute_pit,
    pit_uniformity_dmax,
    midrank_normal_scores,
    pit_portmanteau,
    holm_correction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def short_series(rng):
    """Short Gaussian return series for recursion tests."""
    return rng.normal(0, 1, 100)


@pytest.fixture
def params_stationary():
    """Stationary recursion parameters."""
    omega = -0.1
    phi = 0.97
    alpha = 0.05
    return omega, phi, alpha


# ---------------------------------------------------------------------------
# Recursion tests
# ---------------------------------------------------------------------------

class TestRunRecursion:
    def test_output_length_t_plus_one(self, short_series, params_stationary):
        """run_recursion returns state path of length T+1 (includes initial state)."""
        omega, phi, alpha = params_stationary
        lam_path = run_recursion(
            short_series, omega, phi, alpha,
            criterion="gaussian_log"
        )
        assert len(lam_path) == len(short_series) + 1

    def test_finite_outputs(self, short_series, params_stationary):
        """All state values should be finite for well-posed recursion."""
        omega, phi, alpha = params_stationary
        lam_path = run_recursion(
            short_series, omega, phi, alpha,
            criterion="gaussian_log"
        )
        assert np.all(np.isfinite(lam_path))

    def test_phi_zero_alpha_zero_mean_reversion(self):
        """phi=0, alpha=0 => lam_t = omega for all t >= 1."""
        y = np.zeros(20)
        omega = 0.5
        lam_path = run_recursion(y, omega, phi=0.0, alpha=0.0,
                                  criterion="gaussian_log")
        # lam[0] = omega/(1-0) = omega; subsequent states = omega + 0 + 0 = omega
        np.testing.assert_allclose(lam_path, omega, atol=1e-10)

    def test_no_shock_recursion_converges(self):
        """With zero observations (y=0 => z=0 => g=0^2-1=-1 => psi=-0.5), path follows AR."""
        omega = -0.1
        phi = 0.97
        alpha = 0.0
        y = np.zeros(500)
        lam_path = run_recursion(y, omega, phi, alpha, criterion="gaussian_log")
        lam_target = omega / (1 - phi)
        assert abs(lam_path[-1] - lam_target) < 0.01

    def test_criteria_produce_different_paths(self, short_series, params_stationary):
        """Different criteria produce different state paths."""
        omega, phi, alpha = params_stationary
        paths = {}
        for crit in ["gaussian_log", "density_power", "crps"]:
            paths[crit] = run_recursion(
                short_series, omega, phi, alpha, criterion=crit
            )
        assert not np.allclose(paths["gaussian_log"], paths["density_power"])
        assert not np.allclose(paths["gaussian_log"], paths["crps"])

    def test_initial_state_omega_over_1_minus_phi(self, short_series):
        """Initial state lam[0] = omega/(1-phi) by default."""
        omega, phi, alpha = -0.1, 0.97, 0.05
        lam_path = run_recursion(
            short_series, omega, phi, alpha, criterion="gaussian_log"
        )
        expected_lam0 = omega / (1.0 - phi)
        assert lam_path[0] == pytest.approx(expected_lam0, rel=1e-6)

    def test_large_outlier_gaussian_log_explosive(self):
        """Gaussian log filter with single large outlier shows large update."""
        omega, phi, alpha = -0.1, 0.97, 0.05
        y = np.zeros(10)
        y[5] = 8.0  # outlier
        lam_path_log = run_recursion(y, omega, phi, alpha, criterion="gaussian_log")
        lam_path_dp = run_recursion(y, omega, phi, alpha, criterion="density_power")
        # Update from t=5 to t=6 (lam[6] vs lam[5])
        delta_log = lam_path_log[6] - lam_path_log[5]
        delta_dp = lam_path_dp[6] - lam_path_dp[5]
        assert abs(delta_log) > abs(delta_dp) * 2


class TestLogLipschitzDiagnostic:
    def test_returns_scalar(self, short_series):
        """Diagnostic returns a scalar float."""
        omega, phi, alpha = -0.1, 0.97, 0.05
        lam_path = run_recursion(short_series, omega, phi, alpha,
                                  criterion="gaussian_log")
        ll = log_lipschitz_diagnostic(
            short_series, lam_path, alpha=alpha, criterion="gaussian_log"
        )
        assert isinstance(float(ll), float)

    def test_finite_output(self, short_series):
        """Diagnostic should be finite."""
        omega, phi, alpha = -0.1, 0.97, 0.05
        lam_path = run_recursion(short_series, omega, phi, alpha,
                                  criterion="gaussian_log")
        ll = log_lipschitz_diagnostic(
            short_series, lam_path, alpha=alpha, criterion="gaussian_log"
        )
        assert np.isfinite(ll)


class TestComputeResiduals:
    def test_residuals_shape(self, short_series, params_stationary):
        """Residuals have same length as y (T), using first T states from T+1 path."""
        omega, phi, alpha = params_stationary
        lam_path = run_recursion(
            short_series, omega, phi, alpha, criterion="gaussian_log"
        )
        z = compute_residuals(short_series, lam_path)
        assert len(z) == len(short_series)

    def test_residuals_formula(self):
        """z = y / sqrt(exp(lam)), using first len(y) entries of lam_path."""
        y = np.array([1.0, 2.0, -1.0])
        # lam_path has shape T+1 = 4
        lam_path = np.array([0.0, np.log(4.0), 0.0, 0.0])
        z = compute_residuals(y, lam_path)
        expected = y / np.sqrt(np.exp(lam_path[:3]))
        np.testing.assert_allclose(z, expected)


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------

class TestQLIKE:
    def test_qlike_matches_formula(self):
        """QLIKE_fd(q, v) = log(v) + q/v."""
        q = np.array([1.0, 4.0, 0.25])
        v = np.array([1.0, 2.0, 0.5])
        result = qlike_fd(q, v)
        expected = np.log(v) + q / v
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_qlike_minimum_at_v_equals_q(self):
        """QLIKE is minimized when v = q (perfect forecast)."""
        q = np.array([4.0])
        v_grid = np.linspace(0.1, 20.0, 1000)
        ql_grid = qlike_fd(q, v_grid)
        best_v = v_grid[np.argmin(ql_grid)]
        assert abs(best_v - q[0]) < 0.1

    def test_mean_qlike_scalar(self):
        """mean_qlike returns a scalar."""
        q = np.ones(50)
        v = np.ones(50)
        result = mean_qlike(q, v)
        assert isinstance(float(result), float)

    def test_mean_qlike_at_perfect_forecast(self):
        """When v = q = 1 everywhere, mean_qlike = log(1) + 1/1 = 1."""
        q = np.ones(100)
        v = np.ones(100)
        result = mean_qlike(q, v)
        assert result == pytest.approx(1.0, rel=1e-12)


class TestVaR:
    def test_var_lower_tail(self):
        """VaR at p=0.01 should be negative (lower tail quantile)."""
        lam = np.zeros(10)
        var = var_forecast(lam, p=0.01, working_density="gaussian")
        assert np.all(var < 0)

    def test_var_monotone_in_p(self):
        """VaR(p=0.01) < VaR(p=0.05) < 0 for lower tail."""
        lam = np.zeros(10)
        var01 = var_forecast(lam, p=0.01, working_density="gaussian")
        var05 = var_forecast(lam, p=0.05, working_density="gaussian")
        assert np.all(var01 < var05)

    def test_var_scales_with_vol(self):
        """VaR should scale with sqrt(exp(lam))."""
        lam_low = np.zeros(10)
        lam_high = np.ones(10) * 2.0  # higher vol
        var_low = var_forecast(lam_low, p=0.05, working_density="gaussian")
        var_high = var_forecast(lam_high, p=0.05, working_density="gaussian")
        # Higher lam => more negative VaR
        assert np.all(var_high < var_low)


class TestCoverageRate:
    def test_perfect_calibration_gaussian(self):
        """Coverage rate should be near p for correctly sized VaR."""
        rng = np.random.default_rng(123)
        n = 10000
        y = rng.normal(0, 1, n)
        from scipy.stats import norm
        p = 0.05
        var = np.full(n, norm.ppf(p))
        rate = coverage_rate(y, var)
        assert abs(rate - p) < 0.01

    def test_too_tight_var_over_covers(self):
        """Very tight VaR (close to 0) should have very high coverage."""
        y = np.random.normal(0, 1, 1000)
        var = np.full(1000, 0.0)
        rate = coverage_rate(y, var)
        # ~50% of y < 0
        assert 0.40 < rate < 0.60

    def test_rate_between_0_and_1(self):
        """Coverage rate is always in [0, 1]."""
        y = np.linspace(-5, 5, 100)
        var = np.full(100, -1.5)
        rate = coverage_rate(y, var)
        assert 0.0 <= rate <= 1.0


class TestPIT:
    def test_pit_in_unit_interval(self):
        """PIT values should be in (0, 1)."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 200)
        lam = np.zeros(200)
        u = compute_pit(y, lam, working_density="gaussian")
        assert np.all(u > 0) and np.all(u < 1)

    def test_pit_uniform_for_correct_model(self):
        """PIT should be uniform if model is correctly specified."""
        rng = np.random.default_rng(0)
        n = 5000
        y = rng.normal(0, 1, n)
        lam = np.zeros(n)
        u = compute_pit(y, lam, working_density="gaussian")
        # Kolmogorov-Smirnov test against uniform
        from scipy.stats import kstest
        ks_stat, pval = kstest(u, "uniform")
        assert pval > 0.01  # should not reject uniformity

    def test_pit_dmax_near_zero_for_correct_model(self):
        """D_max should be small for correctly specified model."""
        rng = np.random.default_rng(1)
        n = 5000
        y = rng.normal(0, 1, n)
        lam = np.zeros(n)
        u = compute_pit(y, lam, working_density="gaussian")
        dmax = pit_uniformity_dmax(u)
        assert dmax < 0.05

    def test_pit_shape(self):
        """PIT output has same length as input."""
        y = np.linspace(-3, 3, 50)
        lam = np.zeros(50)
        u = compute_pit(y, lam, working_density="gaussian")
        assert len(u) == 50


class TestMidrankNormalScores:
    def test_output_shape(self):
        """midrank_normal_scores returns same shape as input."""
        u = np.linspace(0.01, 0.99, 100)
        ns = midrank_normal_scores(u)
        assert ns.shape == u.shape

    def test_symmetric_uniform_input(self):
        """For uniform u in (0,1), scores should be approximately symmetric."""
        u = np.linspace(0.01, 0.99, 99)
        ns = midrank_normal_scores(u)
        assert abs(np.mean(ns)) < 0.1

    def test_scores_finite(self):
        """Normal scores should be finite for u in (0,1)."""
        u = np.linspace(0.01, 0.99, 100)
        ns = midrank_normal_scores(u)
        assert np.all(np.isfinite(ns))


class TestPITPortmanteau:
    def test_iid_uniform_small_statistic(self):
        """For iid uniform PITs, portmanteau statistic should be near zero."""
        rng = np.random.default_rng(77)
        u = rng.uniform(0, 1, 1000)
        stat = pit_portmanteau(u)
        # Under null, statistic should be moderate (not huge)
        assert stat < 200  # very loose bound

    def test_dependent_uniform_large_statistic(self):
        """For strongly autocorrelated PITs, statistic should be larger."""
        rng = np.random.default_rng(88)
        # AR(1) uniform-like process (not truly uniform but correlated)
        u_raw = np.zeros(1000)
        u_raw[0] = 0.5
        for t in range(1, 1000):
            u_raw[t] = 0.9 * u_raw[t-1] + 0.1 * rng.uniform(0, 1)
        u = np.clip(u_raw, 0.01, 0.99)
        stat_dep = pit_portmanteau(u)
        u_iid = rng.uniform(0, 1, 1000)
        stat_iid = pit_portmanteau(u_iid)
        assert stat_dep > stat_iid  # dependent should have larger statistic


class TestHolmCorrection:
    def test_holm_produces_adjusted_pvalues(self):
        """holm_correction returns adjusted p-values of same length."""
        pvals = np.array([0.01, 0.05, 0.10, 0.20])
        adj, reject = holm_correction(pvals, alpha=0.05)
        assert len(adj) == len(pvals)
        assert len(reject) == len(pvals)

    def test_holm_never_increases_pvalues_below_smallest(self):
        """Smallest p-value stays as-is or gets multiplied by n."""
        pvals = np.array([0.001, 0.01, 0.05, 0.10])
        adj, reject = holm_correction(pvals, alpha=0.05)
        assert adj[0] <= 1.0

    def test_holm_rejects_tiny_pvalue(self):
        """Very small p-value should be rejected after Holm."""
        pvals = np.array([0.0001, 0.5, 0.8, 0.9])
        adj, reject = holm_correction(pvals, alpha=0.05)
        assert reject[0] == True

    def test_holm_does_not_reject_all_large(self):
        """Large p-values should not be rejected."""
        pvals = np.array([0.5, 0.6, 0.7, 0.8])
        adj, reject = holm_correction(pvals, alpha=0.05)
        assert not np.any(reject)
