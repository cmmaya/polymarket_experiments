# Experiment C — Identifiability Regimes and Failure Diagnostics

This experiment characterizes regimes where outcome inference becomes ill-posed using market-level diagnostics computed from price–volume histories. We compute (i) a separation proxy (empirical KL-gap proxy), (ii) volatility of log-odds increments ΔX_t, (iii) activity/volume measures, and (iv) a liquidity proxy based on price impact per unit volume. We then regress inferential quality (log score) on these diagnostics and visualize threshold-like degradation in performance.

## Artifacts
- Market diagnostics: `reports/experiment_C_market_diagnostics.csv`
- Regression summary: `reports/experiment_C_regression_summary.csv`
- Threshold sweep: `reports/experiment_C_threshold_sweep.csv`
- Plots:
  - `plots/experiment_C_scatter_quality_vs_sep_proxy.png`
  - `plots/experiment_C_scatter_quality_vs_vol_dx.png`
  - `plots/experiment_C_threshold_sweep_sep_proxy.png`
  - `plots/experiment_C_regression_coeffs.png`

## Regression coefficients (absolute-sorted)

| feature         |         coef |
|:----------------|-------------:|
| vol_dx          |  0.479166    |
| sep_proxy       | -0.0056685   |
| v_mean          | -5.08207e-06 |
| n_obs           | -3.40711e-07 |
| liq_proxy       |  2.59559e-07 |
| v_sum           |  1.20543e-09 |
| trades_per_hour | -2.91571e-10 |
| v_median        |  4.84046e-13 |
