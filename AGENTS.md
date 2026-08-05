# Repository Knowledge Base — Scoring-Rule Volatility Filters

Persistent context for AI agents working in this repository.

---

## Project Identity

**Domain:** Quantitative finance — observation-driven volatility filters  
**Core idea:** The scoring criterion (log, density-power, CRPS, MMD, Student-log) is a design lever separate from the working density. It changes the update driver shape, local risk geometry (J_S, K_S), pseudo-true target, and outlier robustness.  
**Scale-state parametrisation:** `v = exp(λ)`, residual `z = y / √v`, recursion `λ_{t+1} = ω + φ λ_t + α u_S(y_t, λ_t)`.

---

## Module Map

| File | What it contains |
|---|---|
| `src/score_drivers.py` | Score functions, raw drivers `ψ_S`, implemented updates `u_S`, tail-envelope computation for all 5 criteria |
| `src/geometry.py` | Local geometry: `J_S = R''_S(λ†)` and `K_S = E[ψ_S²]` via Gauss-Hermite quadrature; centred diffusion coefficient tuple |
| `src/quadrature.py` | `gauss_hermite_normal_expectation(f, n_nodes)` — integrates `E[f(Z)]` for `Z∼N(0,1)` correctly via `(1/√π) Σ w_i f(√2 x_i)` |
| `src/recursion.py` | `run_recursion(y, omega, phi, alpha, criterion, v_init)` — pure-Python, differentiable via JAX if needed |
| `src/fast_filter.py` | Numba-JIT path: `filter_path()`, `filter_score()`, `warm_jit()` — used inside estimation for speed |
| `src/estimation.py` | `MinScoringRiskEstimator` — L-BFGS-B, 8 multistarts, full gate framework, sandwich-covariance readiness |
| `src/bootstrap.py` | `stationary_block_bootstrap_dates()`, `PanelBootstrap`, `mcs_90()`, `holm_adjust()` |
| `src/diagnostics.py` | `var_coverage()`, `pit_marginal()`, `pit_dynamic()` — portmanteau over lags 1–10 |
| `src/data_utils.py` | `fetch_equity_panel()` (Yahoo Finance, 12 tickers), `fetch_fred_series()`, `compute_range_proxies()` |

---

## Implemented Criteria

### Gaussian Log
```python
# g_log(z) = z² - 1;  J = 0.5,  K = 0.5
# S = 0.5λ + 0.5 exp(-λ) y²;  scaling G_log = 2
```

### Density-Power (β=0.15)
```python
# c_β = (2π)^{-β/2};  κ_β = β c_β (1+β)^{-1/2}
# g_β(z) = (1+β) c_β exp(-β z²/2)(z² - 1) + κ_β
# J ≈ 0.3572056, K ≈ 0.2685
```

### Gaussian CRPS
```python
# g_CRPS(z) = π^{-1/2} - √(2/π) exp(-z²/2)
# J ≈ 0.07052, K ≈ 0.01231
```

### Gaussian MMD-RBF (ℓ=2)
```python
# r_λ = ℓ exp(-λ/2);  D_λ = r_λ² + 1
# u_MMD(λ,z) = 2(r_λ/√D_λ) exp(-z²/(2D_λ)) [z²/D_λ² - 1/D_λ]
#             + 2 r_λ / (r_λ²+2)^{3/2}
# J ≈ 0.03402, K ≈ 0.00246; bandwidth ℓ frozen from training data
```

### Student-t₆ Log
```python
# g_t6(z) = (ν+1) z² / ((ν-2) + z²) - 1,  ν=6
# J = 1/3, K = 1/3 (Bartlett identity; J=K)
```

---

## Gauss-Hermite Quadrature Convention

GH integrates against `exp(-x²)`. For standard normal expectations:

```python
# E[f(Z)] = (1/√π) Σ_i w_i f(√2 x_i)
x, w = np.polynomial.hermite.hermgauss(n_nodes)
E = np.sum(w * f(np.sqrt(2) * x)) / np.sqrt(np.pi)
```

Use 64 nodes as default; cross-check with 128; stability threshold is 1e-8.

---

## Estimation Framework Key Points

- Recursion initial state: `λ̂_1 = ω / (1 − φ)`
- Objective: `(1/T) Σ_t S(P_{λ̂_t}, y_t)` — **same S** drives update AND defines objective
- Optimiser: `scipy.optimize.minimize(method='L-BFGS-B')`, 8 multistarts, retain lowest finite + gated objective
- MMD bandwidth: median pairwise absolute distance on ≤1024 evenly spaced training obs, computed once, frozen
- Gates: grad norm ≤1e-3, min Hessian eigenvalue >1e-8, condition number ≤1e10, log-Lipschitz <0, zero projection hits

---

## Data Conventions

- **Returns:** close-to-close log-returns **in percent**
- **Demeaning:** training-sample mean only, frozen for test period
- **Equity tickers:** `^GSPC, ^IXIC, ^DJI, ^FTSE, ^GDAXI, ^FCHI, ^STOXX50E, ^N225, ^HSI, ^GSPTSE, ^AXJO, ^BVSP`
- **Train/test split:** first 70% training, last 30% test, per-market
- **Range proxies:** Parkinson and Garman-Klass from OHLC, for sensitivity
- **FRED Thai Baht:** `DEXTHUS` series

---

## Bootstrap and Inference

- **Stationary-block bootstrap:** geometric block lengths, common date-axis resampling across 12 markets to preserve cross-market dependence
- **MCS:** Hansen-Lunde-Nason 90%, custom implementation — no off-the-shelf Python package for this exact panel setup
- **Holm correction:** `statsmodels.stats.multitest.multipletests(method='holm')` or custom
- **Equal-market estimand:** always aggregate market-level summaries first, then average across markets — never pool all observations

---

## Numba Fast Filter

- Located in `src/fast_filter.py`; call `warm_jit()` at module load to pre-compile
- `filter_path(y, omega, phi, alpha, params_arr, criterion_id)` — returns state path
- `filter_score(y, omega, phi, alpha, params_arr, criterion_id)` — returns scalar total score
- `criterion_id` integers: 0=log, 1=density_power, 2=crps, 3=mmd, 4=student_log

---

## Test Suite

- **195 tests, 0 failures**
- Run: `pytest tests/ -q`
- Key tests: `test_score_drivers.py` (driver math), `test_geometry.py` (J/K values), `test_quadrature.py` (64/128 node stability), `test_recursion.py` (filter path), `test_bootstrap.py` (MCS, Holm), `test_experiments.py` (end-to-end smoke tests)

---

## Results Artefacts

All in `results/`:

| File | Experiment |
|---|---|
| `exp1_catalogue.csv` | Geometry J/K table |
| `exp1_drivers.png`, `exp1_tail_envelope.png` | Driver shape plots |
| `exp2_results.csv`, `exp2_pseudo_true.png` | Pseudo-true target curves |
| `exp3_table.csv`, `exp3_impulse.png` | Impulse response |
| `exp4_table.csv`, `exp4_paths.png` | MC contamination |
| `exp5a_coefficients.csv`, `exp5b_mean_flow.csv`, `exp5c_ou_comparison.csv` | HF mesh |
| `exp6_fitted_params.csv` | Fitted parameters per market-cell |
| `exp7_qlike_table.csv`, `exp7_market_results.csv`, `exp7_coverage.csv` | 12-market panel |
| `exp8_bootstrap_summary.csv`, `exp8_mcs_sensitivity.csv`, `exp8_paired_tests.csv` | MCS + bootstrap |
| `exp9_coverage.csv`, `exp9_pit.csv` | VaR + PIT |
| `exp10_tracking.csv` | Tail-stress tracking correlation |
| `exp11a_jump.csv`, `exp11b_break.csv` | Jump stress + break |
| `exp12_summary.csv` | Event studies |
| `RESULTS.md` | Full quantitative summary |

---

## Archive-Dependent Experiments

EXP 4, EXP 10, and EXP 11A require the author-archived latent-state DGP for exact replication. A fallback GARCH(1,1) DGP is implemented and clearly labelled as approximate. Do not claim exact replication for these without the archive.

---

## Extending the Codebase

- **Add a new criterion:** add to `src/score_drivers.py` (driver function + score function), update `geometry.py` to add its GH expectation, add its `criterion_id` int to `fast_filter.py`.
- **Add a new market:** add its ticker to `DATA_TICKERS` list in `src/data_utils.py`.
- **Change bootstrap settings:** pass `n_bootstrap`, `mean_block_length`, and `seed` to `PanelBootstrap` constructor.
