# Experiment D — Stability Under Realistic Perturbations

This experiment evaluates stability of inference under realistic perturbations of the observed history. For each market we define H as the truncated history at cutoff_fraction=0.90, construct perturbed histories H′, compute π(H), π(H′), and report distributions of |π(H)-π(H′)| and |logBF(H)-logBF(H′)|. Results are stratified by activity level (volume quantiles) and the truncation event E_R (whether leading null price rows were removed).

## Outputs
- `reports/experiment_D_deltas.csv` (per-market deltas)
- `reports/experiment_D_summary.csv` (stratified summary)
- `plots/experiment_D_hist_logBF_*.png`, `plots/experiment_D_hist_pi_*.png`

## Stratified summary (median and 90th percentile)

| perturbation             | activity_bin                 | E_R   |   n |   delta_pi_median |   delta_pi_q90 |   delta_logBF_median |   delta_logBF_q90 |
|:-------------------------|:-----------------------------|:------|----:|------------------:|---------------:|---------------------:|------------------:|
| downsample_k_10          | (172008.625, 440985.313]     | False |  16 |       0           |    0           |           0          |         0         |
| downsample_k_10          | (440985.313, 1408240527.302] | False |  17 |       0           |    0.0014      |           0          |         0.406871  |
| downsample_k_10          | (565.063, 172008.625]        | False |  17 |       0           |    0.008       |           0          |         0.0567522 |
| downsample_k_2           | (172008.625, 440985.313]     | False |  16 |       0           |    0           |           0          |         0         |
| downsample_k_2           | (440985.313, 1408240527.302] | False |  17 |       0           |    0           |           0          |         0         |
| downsample_k_2           | (565.063, 172008.625]        | False |  17 |       0           |    0           |           0          |         0         |
| downsample_k_5           | (172008.625, 440985.313]     | False |  16 |       0           |    0           |           0          |         0         |
| downsample_k_5           | (440985.313, 1408240527.302] | False |  17 |       0           |    0           |           0          |         0         |
| downsample_k_5           | (565.063, 172008.625]        | False |  17 |       0           |    0           |           0          |         0         |
| logodds_noise_sigma_0.05 | (172008.625, 440985.313]     | False |  16 |       2.62676e-08 |    6.98193e-05 |           0.0223607  |         0.0586208 |
| logodds_noise_sigma_0.05 | (440985.313, 1408240527.302] | False |  17 |       3.50555e-05 |    0.000129602 |           0.0170687  |         0.0632477 |
| logodds_noise_sigma_0.05 | (565.063, 172008.625]        | False |  17 |       9.68734e-08 |    0.00177722  |           0.0192884  |         0.0807786 |
| logodds_noise_sigma_0.1  | (172008.625, 440985.313]     | False |  16 |       6.8006e-09  |    0.000265813 |           0.00675477 |         0.137375  |
| logodds_noise_sigma_0.1  | (440985.313, 1408240527.302] | False |  17 |       6.6948e-05  |    0.00073094  |           0.0445625  |         0.132443  |
| logodds_noise_sigma_0.1  | (565.063, 172008.625]        | False |  17 |       4.43347e-08 |    0.00362386  |           0.0367304  |         0.108616  |
| round_decimals_3         | (172008.625, 440985.313]     | False |  16 |       0           |    0           |           0          |         0         |
| round_decimals_3         | (440985.313, 1408240527.302] | False |  17 |       0           |    0           |           0          |         0         |
| round_decimals_3         | (565.063, 172008.625]        | False |  17 |       0           |    0           |           0          |         0         |
| volume_mult_sigma_0.1    | (172008.625, 440985.313]     | False |  16 |       0           |    0           |           0          |         0         |
| volume_mult_sigma_0.1    | (440985.313, 1408240527.302] | False |  17 |       0           |    0           |           0          |         0         |
| volume_mult_sigma_0.1    | (565.063, 172008.625]        | False |  17 |       0           |    0           |           0          |         0         |
| volume_mult_sigma_0.25   | (172008.625, 440985.313]     | False |  16 |       0           |    0           |           0          |         0         |
| volume_mult_sigma_0.25   | (440985.313, 1408240527.302] | False |  17 |       0           |    0           |           0          |         0         |
| volume_mult_sigma_0.25   | (565.063, 172008.625]        | False |  17 |       0           |    0           |           0          |         0         |

## Notes
- logBF is computed from π via log-odds (prior odds = 1). Replace logBF_from_pi if your inference engine outputs BF directly.
- Replace posterior_model_predict(...) with your Bayesian posterior π_t when ready.
