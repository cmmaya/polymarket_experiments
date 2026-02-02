# Experiment A — Backtest Outcome Inference Across Resolved Markets

This experiment evaluates outcome inference quality across a universe of resolved binary markets. For each market, we construct truncated histories at multiple cutoffs (fractions of the available history length) and compute predicted probabilities of the resolved label using: (i) the raw market price (YES probability), (ii) a smoothed price baseline, (iii) a simple logistic regression trained on summary features, and (iv) a placeholder hook for the Bayesian posterior π_t (to be replaced by the inverse-problem inference model). Predictions are scored against the audited resolution labels using Brier score, log score, AUC, and calibration metrics (reliability diagrams and ECE).

## Aggregated metrics by cutoff and model
|   cutoff_fraction | model          |      auc |     brier |       ece |   log_score |   n_samples |
|------------------:|:---------------|---------:|----------:|----------:|------------:|------------:|
|              0.1  | logreg         | 0.804825 | 0.236595  | 0.308868  |   0.672436  |          50 |
|              0.1  | posterior_hook | 0.913377 | 0.0949655 | 0.12448   |   0.313522  |          50 |
|              0.1  | raw_price      | 0.913377 | 0.0949655 | 0.12448   |   0.313522  |          50 |
|              0.1  | smoothed_price | 0.914474 | 0.0927316 | 0.138216  |   0.309232  |          50 |
|              0.25 | logreg         | 0.804825 | 0.180884  | 0.200515  |   0.539245  |          50 |
|              0.25 | posterior_hook | 0.982456 | 0.0526895 | 0.10394   |   0.199463  |          50 |
|              0.25 | raw_price      | 0.982456 | 0.0526895 | 0.10394   |   0.199463  |          50 |
|              0.25 | smoothed_price | 0.982456 | 0.0524863 | 0.104703  |   0.199406  |          50 |
|              0.5  | logreg         | 0.813596 | 0.145795  | 0.14999   |   0.447916  |          50 |
|              0.5  | posterior_hook | 0.985746 | 0.0485795 | 0.0820601 |   0.173282  |          50 |
|              0.5  | raw_price      | 0.985746 | 0.0485795 | 0.0820601 |   0.173282  |          50 |
|              0.5  | smoothed_price | 0.986842 | 0.0473801 | 0.0817206 |   0.170205  |          50 |
|              0.75 | logreg         | 0.899123 | 0.108624  | 0.0831111 |   0.340541  |          50 |
|              0.75 | posterior_hook | 1        | 0.0165172 | 0.0447803 |   0.0614776 |          50 |
|              0.75 | raw_price      | 1        | 0.0165172 | 0.0447803 |   0.0614776 |          50 |
|              0.75 | smoothed_price | 1        | 0.0168875 | 0.0451505 |   0.0625767 |          50 |
|              0.9  | logreg         | 0.958333 | 0.0823135 | 0.109075  |   0.27444   |          50 |
|              0.9  | posterior_hook | 1        | 0.0176109 | 0.0282205 |   0.0563153 |          50 |
|              0.9  | raw_price      | 1        | 0.0176109 | 0.0282205 |   0.0563153 |          50 |
|              0.9  | smoothed_price | 1        | 0.0171396 | 0.0281923 |   0.0539313 |          50 |
|              0.95 | logreg         | 0.934211 | 0.0996993 | 0.137647  |   0.315668  |          50 |
|              0.95 | posterior_hook | 1        | 0.0168787 | 0.0244607 |   0.050738  |          50 |
|              0.95 | raw_price      | 1        | 0.0168787 | 0.0244607 |   0.050738  |          50 |
|              0.95 | smoothed_price | 1        | 0.0153071 | 0.0255571 |   0.0452069 |          50 |

## Pooled metrics across all cutoffs
| model          |     brier |   log_score |      auc |       ece |   n_samples |
|:---------------|----------:|------------:|---------:|----------:|------------:|
| logreg         | 0.142318  |    0.431708 | 0.84576  | 0.122688  |         300 |
| posterior_hook | 0.0412069 |    0.142466 | 0.985532 | 0.0398903 |         300 |
| raw_price      | 0.0412069 |    0.142466 | 0.985532 | 0.0398903 |         300 |
| smoothed_price | 0.040322  |    0.140093 | 0.987086 | 0.0417445 |         300 |

## Notes on artifacts
- Reliability diagrams are saved under `plots/` for each cutoff and model.
- A tabular summary of metrics is also saved as an image under `plots/`.
