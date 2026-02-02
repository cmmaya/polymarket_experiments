# Experiment B — Posterior Concentration vs Empirical KL-Gap Proxy

This experiment tests whether larger empirical separation between outcomes is associated with faster posterior contraction. For each market, we compute a per-market empirical KL-gap proxy and generate a concentration curve defined as log(1-π_t) when the resolved label is YES (Y=1) and log(π_t) when the resolved label is NO (Y=0), where π_t is the inferred probability of YES given the history up to t. We then compare contraction slopes across bins of the empirical proxy.

## Proxy definition (empirical KL-gap proxy)
| proxy_bin        |
|:-----------------|
| (0.647, 13.781]  |
| (13.781, 22.341] |
| (22.341, 31.669] |
| (31.669, 59.113] |

## Slope summary by proxy bin
| proxy_bin        |   n |   slope_median |   slope_q25 |   slope_q75 |
|:-----------------|----:|---------------:|------------:|------------:|
| (0.647, 13.781]  |  13 |       -6.71146 |    -8.75495 |    -1.95665 |
| (13.781, 22.341] |  12 |       -8.58369 |   -11.4566  |    -6.54351 |
| (22.341, 31.669] |  12 |      -10.5881  |   -12.1212  |    -7.74388 |
| (31.669, 59.113] |  13 |       -9.45398 |   -14.3441  |    -4.89709 |

## Outputs
- plots/experiment_B_concentration_by_bin.png
- plots/experiment_B_slopes_by_bin.png
