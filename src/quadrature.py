"""
Gauss-Hermite and Gauss-Legendre quadrature utilities.

Gauss-Hermite integrates against exp(-x^2), so to compute E[f(Z)] for Z ~ N(0,1):
    E[f(Z)] = (1/sqrt(pi)) * sum_i w_i * f(sqrt(2) * x_i)

# Custom — Context7 found no library equivalent for this exact transformation wrapper
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss
from typing import Callable


def gauss_hermite_nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return n-point Gauss-Hermite nodes and weights for integrating f(x) exp(-x^2).

    Parameters
    ----------
    n : int
        Number of quadrature nodes.

    Returns
    -------
    nodes : np.ndarray shape (n,)
    weights : np.ndarray shape (n,)
    """
    return hermgauss(n)


def expectation_normal(f: Callable[[np.ndarray], np.ndarray], n: int = 64) -> float:
    """
    Compute E[f(Z)] where Z ~ N(0,1) using n-point Gauss-Hermite quadrature.

    The identity is:
        E[f(Z)] = (1/sqrt(pi)) * sum_i w_i * f(sqrt(2) * x_i)

    Parameters
    ----------
    f : callable
        Function of a 1-D array of evaluation points.
    n : int
        Number of quadrature nodes (64 default, 128 for verification).

    Returns
    -------
    float : approximation of E[f(Z)].
    """
    x, w = gauss_hermite_nodes(n)
    z = np.sqrt(2.0) * x
    return float(np.dot(w, f(z)) / np.sqrt(np.pi))


def expectation_mixture(
    f: Callable[[np.ndarray], np.ndarray],
    variances: list[float],
    weights: list[float],
    n: int = 96,
) -> float:
    """
    Compute E[f(Y)] under a Gaussian mixture Y ~ sum_k w_k N(0, sigma_k^2)
    using Gauss-Hermite quadrature per component.

    Parameters
    ----------
    f : callable
        Function accepting an array of Y values.
    variances : list of float
        Component variances [sigma_k^2].
    weights : list of float
        Mixing weights summing to 1.
    n : int
        Number of quadrature nodes per component.

    Returns
    -------
    float : E[f(Y)] under the mixture.
    """
    assert abs(sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"
    x, w = gauss_hermite_nodes(n)
    total = 0.0
    for sigma2, weight in zip(variances, weights):
        sigma = np.sqrt(sigma2)
        y_vals = sigma * np.sqrt(2.0) * x  # Y = sigma * sqrt(2) * x_i
        component = float(np.dot(w, f(y_vals)) / np.sqrt(np.pi))
        total += weight * component
    return total


def expectation_student_standardised(
    f: Callable[[np.ndarray], np.ndarray],
    nu: float,
    n: int = 64,
) -> float:
    """
    Compute E[f(Z)] where Z = T * sqrt((nu-2)/nu), T ~ t_nu,
    using Gauss-Hermite quadrature via a normal mixture representation.

    For the variance-standardised Student-t we use the Gauss-Hermite trick:
    Because a t_nu can be written as Z | V ~ N(0, (nu-2)/V) where V ~ Gamma(nu/2, nu/2),
    we integrate numerically by sampling V from Gamma and Z from conditional normal.
    In practice we use 64-node quadrature over the mixing variable.

    Parameters
    ----------
    f : callable
        Function of standardised Student residuals.
    nu : float
        Degrees of freedom (nu > 2 required).
    n : int
        Number of quadrature nodes.

    Returns
    -------
    float : E[f(Z)] under variance-standardised t_nu.
    """
    from scipy.stats import t as student_t
    import scipy.special as sp

    assert nu > 2, "nu must be > 2 for finite variance"
    # Use Gauss-Hermite after reparameterising the scale mixing integral
    # Alternative: Monte Carlo with deterministic antithetic variates
    # We use a high-precision approach: evaluate at Gaussian quadrature nodes
    # after the probability integral transform
    from scipy.special import roots_legendre
    nodes_gl, w_gl = roots_legendre(n * 4)  # Use more Legendre nodes for accuracy

    # Transform Gauss-Legendre [-1,1] -> (0,1) for CDF inversion
    u = 0.5 * (nodes_gl + 1.0)
    w_gl_transformed = 0.5 * w_gl

    # Compute quantiles of standardised t_nu at these uniform points
    scale = np.sqrt((nu - 2.0) / nu)
    z_vals = student_t.ppf(u, df=nu) * scale  # standardised to unit variance

    result = float(np.dot(w_gl_transformed, f(z_vals)))
    return result
