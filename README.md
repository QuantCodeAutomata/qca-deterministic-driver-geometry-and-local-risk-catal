# Scoring-Rule-Driven Volatility Filters

Reproducible implementation of the deterministic driver geometry, local risk catalogue, and empirical panel study for observation-driven volatility filters parameterised by interchangeable scoring criteria.

## Overview

The scale-state parameterisation `v = exp(λ)`, `z = y / √v` defines the recursion backbone. The core insight studied here is that **criterion choice** — independently of working-density choice — changes:

- The shape and tail behaviour of the implemented update driver
- The local conditional scoring-risk curvature (`J_S`) and innovation variance (`K_S`)
- The pseudo-true target itself under contaminated true laws
- The tail robustness and adaptation speed of the resulting filter

Scoring criteria implemented: **Gaussian log, density-power (β=0.15), Gaussian CRPS, Gaussian MMD-RBF (ℓ=2), Student-t₆ log**.  
Working densities: **Gaussian scale family, variance-standardised Student-t₆ family**.

---

## Repository Structure

```
.
├── data/                   # Data storage (auto-populated by data_utils)
├── src/                    # Core modules
│   ├── score_drivers.py    # Score functions, raw drivers, implemented updates
│   ├── geometry.py         # J_S, K_S geometry via Gauss-Hermite quadrature
│   ├── quadrature.py       # GH quadrature helpers, tail-envelope computation
│   ├── recursion.py        # Observation-driven state recursion engine
│   ├── fast_filter.py      # Numba-accelerated filter path for estimation
│   ├── estimation.py       # Minimum-scoring-risk estimator (L-BFGS-B, 8 multistarts)
│   ├── bootstrap.py        # Stationary-block bootstrap, MCS, Holm correction
│   ├── diagnostics.py      # VaR/PIT/portmanteau diagnostics
│   └── data_utils.py       # Yahoo Finance + FRED data download, preprocessing
├── exp/                    # Experiment scripts
│   ├── exp1_driver_geometry.py       # Deterministic geometry catalogue
│   ├── exp2_pseudo_true_target.py    # Pseudo-true target under contamination
│   ├── exp3_impulse_response.py      # Curvature-normalised outlier impulse
│   ├── exp4_mc_contamination.py      # Monte Carlo robustness study
│   ├── exp5_hf_mesh_refinement.py    # High-frequency mesh refinement
│   ├── exp6_static_estimation.py     # Batch min-scoring-risk estimation
│   ├── exp7_empirical_panel.py       # 12-market factorial panel study
│   ├── exp8_bootstrap_mcs.py         # Panel bootstrap & MCS inference
│   ├── exp9_var_pit.py               # VaR coverage and PIT diagnostics
│   ├── exp10_tail_stress.py          # Heavy-tail tracking stress test
│   ├── exp11_jump_and_break.py       # Jump stress & permanent break
│   └── exp12_event_studies.py        # Thai Baht, SPX, Nile event studies
├── tests/                  # pytest test suite (195 tests)
├── results/                # Generated artefacts (CSVs, PNGs, RESULTS.md)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run tests

```bash
pytest tests/ -q
# 195 passed
```

### 3. Run all experiments

```bash
# Deterministic experiments (no data download needed)
python exp/exp1_driver_geometry.py
python exp/exp2_pseudo_true_target.py
python exp/exp3_impulse_response.py
python exp/exp4_mc_contamination.py
python exp/exp5_hf_mesh_refinement.py

# Empirical experiments (require Yahoo Finance + FRED access)
python exp/exp6_static_estimation.py   # fit parameters; prerequisite for exp7–9
python exp/exp7_empirical_panel.py
python exp/exp8_bootstrap_mcs.py
python exp/exp9_var_pit.py

# Robustness and event studies
python exp/exp10_tail_stress.py
python exp/exp11_jump_and_break.py
python exp/exp12_event_studies.py
```

All outputs are saved to `results/`.

### 4. Environment variable

Market data requires no API key (Yahoo Finance and FRED via `pandas-datareader`).  
No `MASSIVE_API_KEY` is required for this project — all data is fetched from Yahoo Finance or FRED.

---

## Score Drivers Quick Reference

| Criterion | Driver g(z) | Tail | J_S | K_S |
|---|---|---|---|---|
| Gaussian Log | z² − 1 | Unbounded | 0.5000 | 0.5000 |
| Density-Power (β=0.15) | (1+β)c_β exp(−βz²/2)(z²−1) + κ_β | Bounded | 0.3572 | 0.2685 |
| Gaussian CRPS | π⁻¹/² − √(2/π)exp(−z²/2) | Bounded | 0.0705 | 0.0123 |
| Gaussian MMD (ℓ=2) | state-dependent | Bounded | 0.0340 | 0.0025 |
| Student-t₆ Log | (ν+1)z²/((ν−2)+z²) − 1 | Bounded | 0.3333 | 0.3333 |

---

## Key Results

See [`results/RESULTS.md`](results/RESULTS.md) for the full quantitative summary.

**Top-line findings:**
1. J_S = K_S only for the log-score (Bartlett identity); all other criteria have J_S ≠ K_S.
2. Bounded drivers limit outlier propagation 7–30× vs. Gaussian log at shock B=8.
3. Contaminated RMSE: Gaussian log ≈ 3.65 vs. bounded criteria ≈ 0.47–0.70.
4. No single criterion universally dominates: all 5 scoring-rule cells remain in the 90% MCS.
5. Student-t₆ working density achieves best 1% VaR coverage (~1.44% vs. nominal 1%).
6. Robust criteria adapt 2× more slowly to genuine variance breaks — the robustness–adaptability trade-off.

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy, scipy | Numerics, Gauss-Hermite quadrature, optimisation |
| pandas | Data handling, results tabulation |
| matplotlib, seaborn | Visualisation |
| numba | JIT-accelerated filter recursion |
| arch | Benchmark GARCH models |
| yfinance | Equity index data (12-market panel) |
| pandas-datareader | FRED data (Thai Baht series) |
| statsmodels | Holm multiple-testing correction |
| pytest | Test suite |

---

## Limitations and Approximate Experiments

Some experiments note **archive-dependent DGP** for the exact latent-state generator:
- **EXP 4, EXP 10, EXP 11A:** Approximate fallback GARCH(1,1) DGP is used and clearly labelled. Qualitative patterns fully reproduced; exact numerical values may differ from paper tables.
- **EXP 12 (Nile):** Location-branch recursion details are archive-dependent; approximate implementation provided.

---

## Citation

This repository implements the methodology from:

> *Score-Driven Volatility Filters with Interchangeable Scoring Criteria*  
> (Archive reference to be confirmed)

All custom implementations reference the specific paper equations in source comments.
