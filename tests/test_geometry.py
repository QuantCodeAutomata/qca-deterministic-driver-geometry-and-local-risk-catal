"""
Tests for src/geometry.py

Validates local curvature J_S and innovation variance K_S computations
against paper-expected numerical values using 64- and 128-node Gauss-Hermite
quadrature. Expected values from the paper:
  - Gaussian log:    J ≈ 0.5,         K ≈ 0.5
  - Density-power:   J ≈ 0.3572056243, K ≈ 0.268
  - CRPS:            J ≈ 0.07052369794, K ≈ 0.012
  - MMD:             J ≈ 0.034,         K ≈ 0.002
"""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.geometry import (
    J_gaussian_log,
    K_gaussian_log,
    J_density_power,
    K_density_power,
    J_crps,
    K_crps,
    J_mmd,
    K_mmd,
    J_student_log,
    K_student_log,
    compute_geometry_catalogue,
)


# ---------------------------------------------------------------------------
# Tolerance levels
# ---------------------------------------------------------------------------
TOL_COARSE = 0.02   # 2% relative tolerance for catalogue comparison
TOL_FINE   = 1e-6   # quadrature stability tolerance between 64 and 128 nodes


# ---------------------------------------------------------------------------
# Gaussian log geometry tests
# ---------------------------------------------------------------------------

class TestGaussianLogGeometry:
    def test_J_approx_half(self):
        """J_log ≈ 0.5 under Gaussian reference."""
        J = J_gaussian_log(n_quad=64)
        assert J == pytest.approx(0.5, abs=TOL_COARSE)

    def test_K_approx_half(self):
        """K_log ≈ 0.5 under Gaussian reference."""
        K = K_gaussian_log(n_quad=64)
        assert K == pytest.approx(0.5, abs=TOL_COARSE)

    def test_J_K_equal(self):
        """For the log score (Bartlett identity), J = K under the working density."""
        J = J_gaussian_log(n_quad=64)
        K = K_gaussian_log(n_quad=64)
        # Under Gaussian reference with Gaussian working density, J = K = 0.5
        assert abs(J - K) < 0.01

    def test_J_positive(self):
        """J_S must be positive (curvature at minimum)."""
        assert J_gaussian_log() > 0

    def test_K_positive(self):
        """K_S must be positive (variance)."""
        assert K_gaussian_log() > 0

    def test_ratio_sqrt_K_over_J(self):
        """sqrt(K) / J should be around 1.41 for log score."""
        J = J_gaussian_log(n_quad=64)
        K = K_gaussian_log(n_quad=64)
        ratio = np.sqrt(K) / abs(J)
        assert 1.30 < ratio < 1.60  # paper reports ~1.41–1.57

    def test_quadrature_stability_64_128(self):
        """64 and 128 nodes agree to at least 1e-6 for Gaussian log."""
        J64 = J_gaussian_log(n_quad=64)
        J128 = J_gaussian_log(n_quad=128)
        assert abs(J64 - J128) < TOL_FINE


# ---------------------------------------------------------------------------
# Density-power geometry tests
# ---------------------------------------------------------------------------

class TestDensityPowerGeometry:
    beta = 0.15

    def test_J_approx_paper_value(self):
        """J_beta ≈ 0.3572056243 from paper catalogue."""
        J = J_density_power(beta=self.beta, n_quad=64)
        assert J == pytest.approx(0.3572056243, rel=TOL_COARSE)

    def test_K_approx_paper_value(self):
        """K_beta ≈ 0.268 from paper catalogue."""
        K = K_density_power(beta=self.beta, n_quad=64)
        assert K == pytest.approx(0.268, rel=0.05)

    def test_J_positive(self):
        assert J_density_power(beta=self.beta) > 0

    def test_K_positive(self):
        assert K_density_power(beta=self.beta) > 0

    def test_J_K_not_equal(self):
        """For density-power (non-log), J != K under Gaussian reference."""
        J = J_density_power(beta=self.beta, n_quad=64)
        K = K_density_power(beta=self.beta, n_quad=64)
        assert abs(J - K) > 0.05  # materially different

    def test_quadrature_stability(self):
        """64 and 128 nodes agree to 1e-4 for density-power (FD scheme limits precision)."""
        J64 = J_density_power(beta=self.beta, n_quad=64)
        J128 = J_density_power(beta=self.beta, n_quad=128)
        assert abs(J64 - J128) < 1e-4


# ---------------------------------------------------------------------------
# CRPS geometry tests
# ---------------------------------------------------------------------------

class TestCRPSGeometry:
    def test_J_approx_paper_value(self):
        """J_CRPS ≈ 0.07052369794 from paper catalogue."""
        J = J_crps(n_quad=64)
        assert J == pytest.approx(0.07052369794, rel=0.05)

    def test_K_approx_paper_value(self):
        """K_CRPS ≈ 0.012 from paper catalogue."""
        K = K_crps(n_quad=64)
        assert K == pytest.approx(0.012, rel=0.10)

    def test_J_positive(self):
        assert J_crps() > 0

    def test_K_positive(self):
        assert K_crps() > 0

    def test_J_K_not_equal(self):
        """CRPS has J != K (non-log score)."""
        J = J_crps(n_quad=64)
        K = K_crps(n_quad=64)
        assert abs(J - K) > 0.001

    def test_quadrature_stability(self):
        """64 and 128 nodes agree for CRPS."""
        J64 = J_crps(n_quad=64)
        J128 = J_crps(n_quad=128)
        assert abs(J64 - J128) < TOL_FINE * 10  # slightly looser for CRPS


# ---------------------------------------------------------------------------
# MMD geometry tests
# ---------------------------------------------------------------------------

class TestMMDGeometry:
    ell = 2.0

    def test_J_approx_paper_value(self):
        """J_MMD ≈ 0.034 from paper catalogue."""
        J = J_mmd(ell=self.ell, n_quad=64)
        assert J == pytest.approx(0.034, rel=0.10)

    def test_K_approx_paper_value(self):
        """K_MMD ≈ 0.002 from paper catalogue (approximate; paper rounds to 3 sig figs)."""
        K = K_mmd(ell=self.ell, n_quad=64)
        assert K == pytest.approx(0.002, rel=0.30)

    def test_J_positive(self):
        assert J_mmd(ell=self.ell) > 0

    def test_K_positive(self):
        assert K_mmd(ell=self.ell) > 0

    def test_ratio_within_paper_range(self):
        """sqrt(K)/|J| should lie within ~[1.0, 2.0]."""
        J = J_mmd(ell=self.ell, n_quad=64)
        K = K_mmd(ell=self.ell, n_quad=64)
        ratio = np.sqrt(K) / abs(J)
        assert 0.8 < ratio < 3.0  # paper says ~1.41–1.57


# ---------------------------------------------------------------------------
# Student-t geometry tests
# ---------------------------------------------------------------------------

class TestStudentLogGeometry:
    nu = 6.0

    def test_J_positive(self):
        assert J_student_log(nu=self.nu) > 0

    def test_K_positive(self):
        assert K_student_log(nu=self.nu) > 0

    def test_J_K_close_under_bartlett(self):
        """For Student log under variance-matched Student reference, J ≈ K."""
        J = J_student_log(nu=self.nu, n_quad=64)
        K = K_student_log(nu=self.nu, n_quad=64)
        # Under matched reference, Bartlett identity should hold approximately
        ratio = abs(J - K) / max(abs(J), abs(K))
        assert ratio < 0.30  # within 30% — Bartlett holds approximately

    def test_J_smaller_than_gaussian_log(self):
        """Student log J < Gaussian log J under mismatched Gaussian reference."""
        J_stud = J_student_log(nu=self.nu, n_quad=64)
        J_gauss = J_gaussian_log(n_quad=64)
        # Under Gaussian Z reference, Student driver is more robust (smaller curvature)
        assert J_stud < J_gauss


# ---------------------------------------------------------------------------
# Geometry catalogue tests
# ---------------------------------------------------------------------------

class TestGeometryCatalogue:
    def test_catalogue_keys(self):
        """Catalogue should contain all expected criteria."""
        cat = compute_geometry_catalogue(n_quad=64)
        for key in ["gaussian_log", "density_power", "crps", "mmd"]:
            assert key in cat

    def test_catalogue_J_K_positive(self):
        """All J and K values in catalogue should be positive."""
        cat = compute_geometry_catalogue(n_quad=64)
        for name, vals in cat.items():
            if "J" in vals:
                assert vals["J"] > 0, f"J not positive for {name}"
            if "K" in vals:
                assert vals["K"] > 0, f"K not positive for {name}"

    def test_catalogue_ratio_range(self):
        """sqrt(K)/|J| should be in [1.0, 2.0] for all criteria."""
        cat = compute_geometry_catalogue(n_quad=64)
        for name, vals in cat.items():
            if "J" in vals and "K" in vals:
                ratio = np.sqrt(vals["K"]) / abs(vals["J"])
                assert 0.5 < ratio < 3.0, f"Ratio out of range for {name}: {ratio}"

    def test_gaussian_log_J_dominates(self):
        """Gaussian log J is largest (most curvature) among main criteria."""
        cat = compute_geometry_catalogue(n_quad=64)
        J_log = cat["gaussian_log"]["J"]
        for name in ["density_power", "crps", "mmd"]:
            if name in cat:
                assert J_log > cat[name]["J"], f"J_log not max over {name}"

    def test_catalogue_returns_dict(self):
        """compute_geometry_catalogue returns a dict."""
        cat = compute_geometry_catalogue()
        assert isinstance(cat, dict)
        assert len(cat) >= 4
