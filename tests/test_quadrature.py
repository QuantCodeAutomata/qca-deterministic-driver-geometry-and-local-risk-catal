"""
Tests for src/quadrature.py

Validates Gauss-Hermite quadrature nodes, standard-normal expectations,
mixture expectations, and Student-t expectations.
"""

import numpy as np
import pytest
from scipy.stats import norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quadrature import (
    gauss_hermite_nodes,
    expectation_normal,
    expectation_mixture,
    expectation_student_standardised,
)

# Helper to wrap the mixture API for exp_2-style calls
def _mixture_expectation(f, v0, epsilon, tau, n_quad=96):
    """Wrap expectation_mixture for contaminated Gaussian Q = (1-eps)N(0,v0) + eps*N(0,tau*v0)."""
    variances = [v0, tau * v0]
    weights = [1.0 - epsilon, epsilon]
    return expectation_mixture(f, variances, weights, n=n_quad)


class TestGaussHermiteNodes:
    def test_nodes_64(self):
        """64-node quadrature returns correct shapes."""
        x, w = gauss_hermite_nodes(64)
        assert len(x) == 64
        assert len(w) == 64

    def test_nodes_128(self):
        """128-node quadrature returns correct shapes."""
        x, w = gauss_hermite_nodes(128)
        assert len(x) == 128
        assert len(w) == 128

    def test_weights_positive(self):
        """All weights should be positive."""
        x, w = gauss_hermite_nodes(64)
        assert np.all(w > 0)

    def test_weights_sum_approx_sqrt_pi(self):
        """Gauss-Hermite weights sum to sqrt(pi)."""
        x, w = gauss_hermite_nodes(64)
        assert abs(np.sum(w) - np.sqrt(np.pi)) < 1e-10

    def test_nodes_symmetric(self):
        """Nodes are symmetric about zero."""
        x, w = gauss_hermite_nodes(64)
        x_sorted = np.sort(x)
        assert np.allclose(x_sorted, -x_sorted[::-1], atol=1e-12)

    def test_single_node_raises_or_returns(self):
        """n=1 should work without error."""
        x, w = gauss_hermite_nodes(1)
        assert len(x) == 1


class TestExpectationNormal:
    def test_mean_is_zero(self):
        """E[Z] = 0 for Z ~ N(0,1)."""
        result = expectation_normal(lambda z: z, n=64)
        assert abs(result) < 1e-12

    def test_second_moment_is_one(self):
        """E[Z^2] = 1 for Z ~ N(0,1)."""
        result = expectation_normal(lambda z: z**2, n=64)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_fourth_moment(self):
        """E[Z^4] = 3 for Z ~ N(0,1)."""
        result = expectation_normal(lambda z: z**4, n=64)
        assert result == pytest.approx(3.0, rel=1e-8)

    def test_gaussian_pdf_integrates_to_one(self):
        """E[phi(Z)] should equal phi at origin value related to partition."""
        # E[1] = 1
        result = expectation_normal(lambda z: np.ones_like(z), n=64)
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_even_function_matches_analytic(self):
        """E[exp(-Z^2/2)] = 1/sqrt(2)."""
        result = expectation_normal(lambda z: np.exp(-z**2 / 2), n=64)
        expected = 1.0 / np.sqrt(2.0)
        assert result == pytest.approx(expected, rel=1e-8)

    def test_convergence_64_vs_128(self):
        """E[Z^2 - 1]^2 should agree between 64 and 128 nodes."""
        f = lambda z: (z**2 - 1) ** 2
        r64 = expectation_normal(f, n=64)
        r128 = expectation_normal(f, n=128)
        assert abs(r64 - r128) < 1e-8

    def test_log_driver_expectation_zero(self):
        """E[Z^2 - 1] = 0 under N(0,1) (score unbiasedness)."""
        result = expectation_normal(lambda z: z**2 - 1, n=64)
        assert abs(result) < 1e-10

    def test_psi_second_moment_log_approx_half(self):
        """E[(0.5*(Z^2-1))^2] = E[0.25*(Z^2-1)^2] = 0.25*E[(Z^4-2Z^2+1)] = 0.25*2 = 0.5."""
        result = expectation_normal(lambda z: (0.5 * (z**2 - 1))**2, n=64)
        # E[(Z^4 - 2Z^2 + 1)] = 3 - 2 + 1 = 2, so 0.25 * 2 = 0.5
        assert result == pytest.approx(0.5, rel=1e-8)


class TestExpectationMixture:
    def test_pure_gaussian_matches_normal(self):
        """With epsilon=0, mixture = single Gaussian."""
        result = _mixture_expectation(lambda z: z**2, v0=1.0, epsilon=0.0, tau=4.0, n_quad=64)
        assert result == pytest.approx(1.0, rel=1e-8)

    def test_mixture_second_moment(self):
        """E[Y^2] = (1-eps)*v0 + eps*tau*v0 for Y ~ (1-eps)N(0,v0) + eps*N(0,tau*v0)."""
        v0 = 1.0
        eps = 0.05
        tau = 4.0
        expected = (1 - eps) * v0 + eps * tau * v0
        result = _mixture_expectation(lambda y: y**2, v0=v0, epsilon=eps, tau=tau, n_quad=64)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_mixture_mean_zero(self):
        """E[Y] = 0 for symmetric mixture."""
        result = _mixture_expectation(lambda y: y, v0=1.0, epsilon=0.05, tau=4.0, n_quad=64)
        assert abs(result) < 1e-10

    def test_mixture_convergence_96_vs_128(self):
        """96-node and 128-node quadrature agree for mixture expectations."""
        f = lambda y: y**2 - 1
        r96 = _mixture_expectation(f, v0=1.0, epsilon=0.05, tau=4.0, n_quad=96)
        r128 = _mixture_expectation(f, v0=1.0, epsilon=0.05, tau=4.0, n_quad=128)
        assert abs(r96 - r128) < 1e-8

    def test_mixture_api_with_variances_weights(self):
        """Direct API call with variances and weights lists."""
        variances = [1.0, 4.0]
        weights = [0.95, 0.05]
        expected = 0.95 * 1.0 + 0.05 * 4.0
        result = expectation_mixture(lambda y: y**2, variances=variances, weights=weights, n=96)
        assert result == pytest.approx(expected, rel=1e-6)


class TestExpectationStudentStandardised:
    def test_variance_is_one(self):
        """Variance-standardised t_nu has E[Z^2] = 1."""
        result = expectation_student_standardised(lambda z: z**2, nu=6.0, n=64)
        assert result == pytest.approx(1.0, rel=0.05)

    def test_mean_is_zero(self):
        """E[Z] = 0 for symmetric Student distribution."""
        result = expectation_student_standardised(lambda z: z, nu=6.0, n=64)
        assert abs(result) < 0.01

    def test_nu_large_approaches_normal(self):
        """For large nu, Student approaches Gaussian: E[Z^4] approaches 3."""
        result = expectation_student_standardised(lambda z: z**4, nu=100.0, n=64)
        assert result == pytest.approx(3.0, rel=0.05)
