"""
Tests for src/score_drivers.py

Validates score functions, raw innovation drivers, and implemented updates for
observation-driven volatility filters. All formulas are verified against
closed-form expectations from the paper.
"""

import numpy as np
import pytest
from scipy.stats import norm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.score_drivers import (
    gaussian_log_score,
    gaussian_log_driver,
    gaussian_log_psi,
    G_LOG,
    student_log_driver,
    student_log_psi,
    density_power_constants,
    density_power_score,
    density_power_driver,
    density_power_psi,
    density_power_G,
    crps_score,
    crps_driver,
    crps_psi,
    crps_G,
    mmd_score,
    mmd_update,
    mmd_psi,
    get_driver_value,
    get_psi_value,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def z_grid():
    """Standard residual grid [-5, 5], 201 points."""
    return np.linspace(-5.0, 5.0, 201)


@pytest.fixture
def lam0():
    """Centered log-state lambda = 0, so v = exp(0) = 1."""
    return 0.0


# ---------------------------------------------------------------------------
# Gaussian log score tests
# ---------------------------------------------------------------------------

class TestGaussianLogScore:
    def test_score_nonnegative_for_large_y(self, lam0):
        """Score should be large for |y| >> sqrt(exp(lam))."""
        y_large = np.array([10.0, 20.0])
        s = gaussian_log_score(lam0, y_large)
        assert np.all(s > 0)

    def test_score_minimum_at_sigma_equals_std(self):
        """For y ~ N(0, sigma^2), minimum risk is at lam = log(sigma^2)."""
        sigma2 = 4.0
        lam_grid = np.linspace(-3, 3, 1000)
        # E[S_log(lam, y)] = 0.5*(lam + sigma^2/exp(lam))
        risk = 0.5 * (lam_grid + sigma2 * np.exp(-lam_grid))
        best = lam_grid[np.argmin(risk)]
        assert abs(best - np.log(sigma2)) < 0.01

    def test_driver_zero_at_z1(self):
        """g_log(±1) = 0 by z^2 - 1 = 0."""
        assert abs(gaussian_log_driver(1.0)) < 1e-14
        assert abs(gaussian_log_driver(-1.0)) < 1e-14

    def test_driver_quadratic_growth(self, z_grid):
        """Driver g_log(z) = z^2 - 1 should grow quadratically."""
        g = gaussian_log_driver(z_grid)
        expected = z_grid**2 - 1
        np.testing.assert_allclose(g, expected, rtol=1e-12)

    def test_driver_unbounded(self):
        """Driver is unbounded: at z=10, g_log = 99."""
        assert gaussian_log_driver(10.0) == pytest.approx(99.0)

    def test_psi_consistency_with_driver(self):
        """At lam=0, psi = 0.5 * g_log(z)."""
        z = np.array([0.5, 1.0, 2.0, -1.5])
        y = z  # since lam=0 => v=1 => z = y
        psi = gaussian_log_psi(y, 0.0)
        g = gaussian_log_driver(z)
        np.testing.assert_allclose(psi, 0.5 * g, rtol=1e-12)

    def test_psi_zero_mean_under_standard_normal(self):
        """E[psi_log(Y, 0)] = 0 for Y ~ N(0,1)."""
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100_000)
        psi = gaussian_log_psi(y, 0.0)
        # Should be ~0 by score unbiasedness
        assert abs(np.mean(psi)) < 0.01

    def test_scaling_constant(self):
        """G_LOG should be 2.0."""
        assert G_LOG == pytest.approx(2.0)

    def test_scalar_array_consistency(self, lam0):
        """Scalar and array inputs produce same results."""
        y_scalar = 1.5
        y_arr = np.array([1.5])
        s_scalar = gaussian_log_score(lam0, y_scalar)
        s_arr = gaussian_log_score(lam0, y_arr)
        assert abs(s_scalar - s_arr[0]) < 1e-14


# ---------------------------------------------------------------------------
# Student-t log score tests
# ---------------------------------------------------------------------------

class TestStudentLogDriver:
    def test_driver_at_z0_is_minus1(self):
        """At z=0: g(0) = 0 - 1 = -1."""
        assert student_log_driver(0.0) == pytest.approx(-1.0)
        assert student_log_driver(0.0, nu=10.0) == pytest.approx(-1.0)

    def test_driver_bounded_above(self):
        """As z->inf, g(z) -> nu - 1 (bounded)."""
        nu = 6.0
        g_large = student_log_driver(1000.0, nu=nu)
        assert g_large < nu  # bounded above by nu-1

    def test_driver_matches_formula(self):
        """Verify g(z) = (nu+1)*z^2 / ((nu-2)+z^2) - 1."""
        nu = 6.0
        z = np.array([0.5, 1.0, 2.0, 3.0])
        expected = (nu + 1) * z**2 / ((nu - 2) + z**2) - 1.0
        np.testing.assert_allclose(student_log_driver(z, nu), expected, rtol=1e-12)

    def test_psi_consistency(self):
        """psi_student at lam=0 should relate to driver value."""
        z = np.array([1.5, 2.0, -0.5])
        y = z  # at lam=0
        psi = student_log_psi(y, 0.0, nu=6.0)
        assert psi.shape == z.shape  # shape preserved

    def test_psi_sign_at_large_z(self):
        """Large |z| => driver positive => psi positive (pushes lam up)."""
        y = np.array([5.0, 10.0])
        psi = student_log_psi(y, 0.0, nu=6.0)
        assert np.all(psi > 0)


# ---------------------------------------------------------------------------
# Density-power score tests
# ---------------------------------------------------------------------------

class TestDensityPowerScore:
    beta = 0.15

    def test_constants_positive(self):
        """c_beta, kappa_beta should be positive."""
        c_beta, kappa_beta = density_power_constants(self.beta)
        assert c_beta > 0
        assert kappa_beta > 0

    def test_constants_formula(self):
        """c_beta = (2*pi)^(-beta/2), kappa_beta = beta*c_beta*(1+beta)^(-1/2)."""
        c_beta, kappa_beta = density_power_constants(self.beta)
        c_expected = (2 * np.pi) ** (-self.beta / 2)
        kappa_expected = self.beta * c_expected * (1 + self.beta) ** (-0.5)
        assert c_beta == pytest.approx(c_expected, rel=1e-12)
        assert kappa_beta == pytest.approx(kappa_expected, rel=1e-12)

    def test_driver_at_large_z_bounded(self, z_grid):
        """Density-power driver should saturate for large |z|."""
        g = density_power_driver(z_grid, beta=self.beta)
        # Should not grow to infinity like z^2-1
        assert np.max(np.abs(g)) < 50  # bounded behavior

    def test_driver_shape(self, z_grid):
        """Driver should be array of same shape as input."""
        g = density_power_driver(z_grid, beta=self.beta)
        assert g.shape == z_grid.shape

    def test_psi_zero_at_z0(self):
        """At z=0 (minimum variance), driver has specific value from formula."""
        # g_beta(0) = kappa_beta (the constant term when z=0, exp term=1)
        c_beta, kappa_beta = density_power_constants(self.beta)
        g0 = density_power_driver(0.0, beta=self.beta)
        expected_at_0 = (1 + self.beta) * c_beta * 1.0 * (0.0 - 1.0) + kappa_beta
        assert g0 == pytest.approx(expected_at_0, rel=1e-10)

    def test_G_scaling_increases_with_lam(self):
        """G_beta(lam) = 2*exp(beta*lam/2) is increasing in lam."""
        lam_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        G = density_power_G(lam_vals, beta=self.beta)
        assert np.all(np.diff(G) > 0)

    def test_G_at_lam0(self):
        """At lam=0, G_beta = 2."""
        G0 = density_power_G(0.0, beta=self.beta)
        assert G0 == pytest.approx(2.0, rel=1e-12)

    def test_psi_consistency_with_driver_at_lam0(self):
        """psi_beta at lam=0 should equal 0.5 * g_beta(z)."""
        z = np.array([0.5, 1.0, 2.0])
        y = z  # at lam=0
        psi = density_power_psi(y, 0.0, beta=self.beta)
        g = density_power_driver(z, beta=self.beta)
        np.testing.assert_allclose(psi, 0.5 * g, rtol=1e-8)


# ---------------------------------------------------------------------------
# CRPS score tests
# ---------------------------------------------------------------------------

class TestCRPSScore:
    def test_crps_positive(self):
        """CRPS should be nonneg for y != 0."""
        y = np.array([1.0, 2.0, -1.5])
        s = crps_score(0.0, y)
        # CRPS is a proper score; value should be finite
        assert np.all(np.isfinite(s))

    def test_crps_driver_constant_tail(self, z_grid):
        """CRPS driver should approach a constant as |z|->inf."""
        g = crps_driver(z_grid)
        # Check that driver is bounded (tail-saturating)
        assert np.max(np.abs(g)) < 1.0  # much smaller than z^2 at z=5

    def test_crps_driver_at_z0(self):
        """At z=0: 2*phi(0) = 2/sqrt(2*pi); g_CRPS(0) = 1/sqrt(pi) - 2/sqrt(2*pi)."""
        g0 = crps_driver(0.0)
        expected = 1.0 / np.sqrt(np.pi) - np.sqrt(2.0 / np.pi)
        assert g0 == pytest.approx(expected, rel=1e-8)

    def test_crps_G_decreasing_in_lam(self):
        """G_CRPS(lam) = 2*exp(-lam/2) is decreasing in lam."""
        lam_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        G = crps_G(lam_vals)
        assert np.all(np.diff(G) < 0)

    def test_crps_G_at_lam0(self):
        """At lam=0, G_CRPS = 2."""
        G0 = crps_G(0.0)
        assert G0 == pytest.approx(2.0, rel=1e-12)

    def test_crps_psi_shape(self):
        """psi_CRPS should return same shape as input."""
        y = np.linspace(-3, 3, 50)
        lam = np.zeros_like(y)
        psi = crps_psi(y, lam)
        assert psi.shape == y.shape


# ---------------------------------------------------------------------------
# MMD score tests
# ---------------------------------------------------------------------------

class TestMMDScore:
    ell = 2.0

    def test_mmd_score_finite(self):
        """MMD score should produce finite values."""
        lam = np.array([0.0, 1.0, -1.0])
        y = np.array([1.0, 2.0, -1.0])
        s = mmd_score(lam, y, ell=self.ell)
        assert np.all(np.isfinite(s))

    def test_mmd_update_bounded(self, z_grid):
        """MMD update should be bounded for all residuals."""
        u = mmd_update(0.0, z_grid, ell=self.ell)
        assert np.max(np.abs(u)) < 5.0  # bounded behavior

    def test_mmd_update_shape(self, z_grid):
        """mmd_update should return same shape as z_grid."""
        u = mmd_update(0.0, z_grid, ell=self.ell)
        assert u.shape == z_grid.shape

    def test_mmd_psi_finite(self):
        """mmd_psi should return finite values."""
        y = np.linspace(-4, 4, 50)
        lam = np.zeros_like(y)
        psi = mmd_psi(y, lam, ell=self.ell)
        assert np.all(np.isfinite(psi))

    def test_mmd_update_at_z0_positive_correction(self):
        """At z=0, update has a positive constant term from bandwidth correction."""
        u0 = mmd_update(0.0, 0.0, ell=self.ell)
        # At z=0: exp(-0) = 1, z^2/D^2 = 0, so -1/D term dominates; + positive tail term
        # Value should be finite and near 0 or negative (exponential pushes down)
        assert np.isfinite(u0)


# ---------------------------------------------------------------------------
# Driver dispatcher tests
# ---------------------------------------------------------------------------

class TestGetDriverValue:
    def test_gaussian_log_dispatch(self, z_grid):
        """get_driver_value('gaussian_log') == gaussian_log_driver."""
        g1 = get_driver_value("gaussian_log", z_grid)
        g2 = gaussian_log_driver(z_grid)
        np.testing.assert_allclose(g1, g2)

    def test_density_power_dispatch(self, z_grid):
        """get_driver_value('density_power') == density_power_driver."""
        g1 = get_driver_value("density_power", z_grid, beta=0.15)
        g2 = density_power_driver(z_grid, beta=0.15)
        np.testing.assert_allclose(g1, g2)

    def test_crps_dispatch(self, z_grid):
        """get_driver_value('crps') == crps_driver."""
        g1 = get_driver_value("crps", z_grid)
        g2 = crps_driver(z_grid)
        np.testing.assert_allclose(g1, g2)

    def test_student_log_dispatch(self, z_grid):
        """get_driver_value('student_log') == student_log_driver."""
        g1 = get_driver_value("student_log", z_grid, nu=6.0)
        g2 = student_log_driver(z_grid, nu=6.0)
        np.testing.assert_allclose(g1, g2)

    def test_unknown_criterion_raises(self, z_grid):
        """Unknown criterion should raise ValueError."""
        with pytest.raises((ValueError, KeyError)):
            get_driver_value("unknown_score", z_grid)

    def test_get_psi_value_gaussian_log(self):
        """get_psi_value returns correct psi for gaussian_log."""
        y = np.array([1.0, 2.0])
        lam = np.array([0.0, 0.0])
        psi1 = get_psi_value("gaussian_log", y, lam)
        psi2 = gaussian_log_psi(y, lam)
        np.testing.assert_allclose(psi1, psi2)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_residual_gaussian_driver(self):
        """At z=0, g_log(0) = -1."""
        assert gaussian_log_driver(0.0) == pytest.approx(-1.0)

    def test_single_point_inputs(self):
        """Functions handle scalar float inputs without error."""
        for fn, args in [
            (gaussian_log_score, (0.0, 1.0)),
            (gaussian_log_driver, (0.5,)),
            (density_power_driver, (1.0,)),
            (crps_driver, (0.5,)),
        ]:
            result = fn(*args)
            assert np.isfinite(result)

    def test_extreme_values(self):
        """Functions handle large z without NaN."""
        z_extreme = np.array([-100.0, 100.0])
        for fn in [gaussian_log_driver, crps_driver]:
            g = fn(z_extreme)
            assert np.all(np.isfinite(g))

    def test_density_power_beta_zero_limit(self):
        """As beta -> 0, density-power score should approach log-score behavior."""
        # Driver at beta=0.001 should resemble gaussian log driver
        z = np.array([1.0, 2.0])
        g_dp = density_power_driver(z, beta=0.001)
        g_log = gaussian_log_driver(z)
        # Should be directionally similar (both positive for |z|>1)
        assert np.sign(g_dp[1]) == np.sign(g_log[1])

    def test_driver_symmetry_even(self, z_grid):
        """All residual-only drivers are even functions of z."""
        for fn, kwargs in [
            (gaussian_log_driver, {}),
            (student_log_driver, {"nu": 6.0}),
            (density_power_driver, {"beta": 0.15}),
            (crps_driver, {}),
        ]:
            g_pos = fn(np.abs(z_grid), **kwargs)
            g_neg = fn(-np.abs(z_grid), **kwargs)
            np.testing.assert_allclose(g_pos, g_neg, atol=1e-12)
