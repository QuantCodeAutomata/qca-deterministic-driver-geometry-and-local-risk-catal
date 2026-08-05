"""
Scoring-Rule-Driven Volatility Filters: core implementation package.

Modules:
    quadrature    - Gauss-Hermite and Gauss-Legendre quadrature utilities
    score_drivers - Score functions, raw drivers, and implemented updates
    geometry      - Local geometry J_S, K_S computation via quadrature
    recursion     - Observation-driven state recursion engine
    estimation    - Minimum-scoring-risk batch estimator
    diagnostics   - VaR, PIT, and QLIKE evaluation utilities
    bootstrap     - Panel stationary bootstrap and MCS
    data_utils    - Data loading, preprocessing, and proxy computation
"""
