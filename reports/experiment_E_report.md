# Experiment E — Information Gain Trajectories

This experiment analyzes how information about the binary outcome accumulates over time in market histories. For each market we compute a posterior trajectory π_t from the observed history H_t and define the information gain trajectory as IG(H_t) = KL(Bern(π_t) || Bern(prior)), using a symmetric prior p0=0.5. We aggregate trajectories across markets as a function of normalized time and cumulative volume fraction, and identify information bursts using robust thresholds on ΔIG.

## Artifacts
- `plots/experiment_E_IG_vs_time.png`
- `plots/experiment_E_IG_vs_cumvol.png`
- `plots/experiment_E_bursts_vs_volume.png`
- `reports/experiment_E_market_IG_summary.csv`
- `reports/experiment_E_bursts.csv`
- `reports/experiment_E_bound_by_regime.csv`

## Expected information bound (generic)
- Outcome entropy bound: H(Y) = 0.693147 (nats) for prior p0=0.50.

## Empirical IG_final by activity regime (proxy design regimes)

| regime         |   n |   IG_final_mean |   IG_final_median |   IG_final_q90 |   total_volume_median |   H(Y)_bound |
|:---------------|----:|----------------:|------------------:|---------------:|----------------------:|-------------:|
| activity_bin_1 |  17 |        0.681928 |          0.693132 |       0.693132 |      126262           |     0.693147 |
| activity_bin_2 |  16 |        0.691245 |          0.693132 |       0.693132 |      252848           |     0.693147 |
| activity_bin_3 |  17 |        0.6749   |          0.693132 |       0.693132 |           1.59497e+07 |     0.693147 |

## Notes
- Replace posterior_model_predict(H_t) with your Bayesian inverse-problem posterior π_t to match the paper model.
- If you later define explicit market-design regimes, replace the activity-bin regime assignment with those labels.
