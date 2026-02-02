#!/usr/bin/env python3
"""
Experiment C: Identifiability Regimes and Failure Diagnostics (Market-level)

From experiments.pdf (Experiment C):
- Produce market-level diagnostics:
    * separation proxy (KL gap proxy)
    * volatility of ΔX_t (log-odds increments)
    * activity/volume measures
    * liquidity proxy (if available; with only price+volume we use a simple proxy)
- Run regressions of inferential quality (e.g., log score) against these diagnostics
- Identify regimes where inference degrades abruptly (threshold-like behavior)

This script is fully runnable and uses ONLY your available observables:
- price_yes (probability), volume (takerOnly), timestamp.

Inferential quality target:
- We compute per-market log score using the prediction from Experiment A at a fixed cutoff.
  Default: cutoff_fraction = 0.90 (late but pre-resolution proxy).
  Prediction sources:
    - posterior_hook (replace with your Bayesian π_T later; placeholder is last price_yes)
    - raw_price baseline
    - smoothed baseline
    - logreg baseline (optional; if you already ran Experiment A you can reuse, but here we keep it local)

Outputs:
- plots/experiment_C_scatter_quality_vs_proxy.png
- plots/experiment_C_scatter_quality_vs_volatility.png
- plots/experiment_C_threshold_sweep_proxy.png
- plots/experiment_C_regression_coeffs.png
- reports/experiment_C_report.md
- reports/experiment_C_market_diagnostics.csv
- reports/experiment_C_regression_summary.csv

Notes:
- "ΔX_t" is defined in logit space: X_t = logit(P_t), ΔX_t = X_t - X_{t-1}
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
REPORTS_DIR = Path("reports")

OUTCOMES_PATH_CANDIDATES = [
    Path("data/market_outcomes.csv"),
    Path("data/market_outcomes.tsv"),
    Path("data/market_outcomes.txt"),
    Path("data/market_outcomes"),
]

EPS = 1e-6
N_ECE_BINS = 10

# Use a fixed cutoff to assess "pre-resolution" inferential quality for Experiment C
CUTOFF_FRACTION_FOR_QUALITY = 0.90

# Threshold sweep config for "abrupt degradation" visualization
N_THRESHOLD_POINTS = 30
MIN_MARKETS_PER_SIDE = 5


# -----------------------------
# IO helpers
# -----------------------------
def ensure_dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def find_outcomes_file() -> Path:
    for p in OUTCOMES_PATH_CANDIDATES:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find market outcomes file. Expected one of: "
        + ", ".join(str(p) for p in OUTCOMES_PATH_CANDIDATES)
    )


def load_outcomes(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        try:
            df = pd.read_csv(path)
            if df.shape[1] == 1:
                head = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                if "\t" in head:
                    df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path, sep="\t")

    required = {"name", "resolved_label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Outcomes file must contain columns {required}. Got {list(df.columns)}")

    df = df.copy()
    df["resolved_label"] = df["resolved_label"].astype(str).str.strip().str.upper()
    df = df[df["resolved_label"].isin(["YES", "NO"])].copy()
    df["y_true"] = (df["resolved_label"] == "YES").astype(int)
    df["name"] = df["name"].astype(str).str.strip()
    return df[["name", "resolved_label", "y_true"]]


def load_market_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed_cols = {"timestamp", "price_yes", "price_no", "volume"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    for c in ["price_yes", "price_no", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    first_valid = df["price_yes"].first_valid_index()
    if first_valid is None:
        return df.iloc[0:0].copy()
    df = df.loc[first_valid:].reset_index(drop=True)
    df = df.dropna(subset=["price_yes"]).reset_index(drop=True)
    return df


# -----------------------------
# Math helpers
# -----------------------------
def clip_prob(p: float) -> float:
    return float(np.clip(p, EPS, 1.0 - EPS))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def ewma_last(prices: np.ndarray, alpha: float = 0.2) -> float:
    if len(prices) == 0:
        return float("nan")
    s = float(prices[0])
    for x in prices[1:]:
        s = alpha * float(x) + (1 - alpha) * s
    return float(s)


def build_cutoff_history(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if df.empty:
        return df
    n = len(df)
    k = max(1, int(math.floor(frac * n)))
    return df.iloc[:k].copy()


def posterior_model_predict(df_cut: pd.DataFrame) -> float:
    """
    Same hook as Experiment A/B: replace with Bayesian π_t later.
    Placeholder uses last raw price_yes.
    """
    return float(df_cut["price_yes"].iloc[-1])


# -----------------------------
# Diagnostics (Experiment C)
# -----------------------------
def separation_proxy(df: pd.DataFrame) -> float:
    """
    KL-gap proxy used in Experiment B:
    |mean(X_t) - X_0| / (std(ΔX_t) + eps), X_t=logit(P_t)
    """
    p = df["price_yes"].to_numpy(dtype=float)
    if len(p) < 3:
        return float("nan")
    x = logit(p)
    dx = np.diff(x)
    numer = float(abs(np.mean(x) - x[0]))
    denom = float(np.std(dx, ddof=0) + 1e-8)
    return numer / denom


def volatility_dx(df: pd.DataFrame) -> float:
    """
    Volatility of log-odds increments ΔX_t.
    """
    p = df["price_yes"].to_numpy(dtype=float)
    if len(p) < 3:
        return float("nan")
    x = logit(p)
    dx = np.diff(x)
    return float(np.std(dx, ddof=0))


def activity_volume(df: pd.DataFrame) -> Dict[str, float]:
    """
    Activity diagnostics:
    - total volume, mean volume, median volume
    - n_obs, trades per hour (proxy via timestamps)
    """
    v = df["volume"].to_numpy(dtype=float)
    n = len(df)
    v_sum = float(np.nansum(v))
    v_mean = float(np.nanmean(v)) if n else 0.0
    v_med = float(np.nanmedian(v)) if n else 0.0

    # trades per hour (based on elapsed time in hours)
    t0 = df["timestamp"].iloc[0]
    t1 = df["timestamp"].iloc[-1]
    elapsed_hours = max(1e-6, (t1 - t0).total_seconds() / 3600.0) if pd.notnull(t0) and pd.notnull(t1) else float("nan")
    tph = float(n / elapsed_hours) if np.isfinite(elapsed_hours) else float("nan")

    return {
        "n_obs": float(n),
        "v_sum": v_sum,
        "v_mean": v_mean,
        "v_median": v_med,
        "trades_per_hour": tph,
    }


def liquidity_proxy(df: pd.DataFrame) -> float:
    """
    Simple liquidity proxy from price+volume only:
    - "price impact per unit volume" proxy: median(|ΔP| / (V + eps))
    Lower is "more liquid", higher is "less liquid / higher impact".
    """
    p = df["price_yes"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    if len(p) < 3:
        return float("nan")
    dp = np.abs(np.diff(p))
    vv = v[1:]  # align with diffs
    impact = dp / (vv + 1e-8)
    return float(np.nanmedian(impact))


# -----------------------------
# Quality target (log score)
# -----------------------------
def market_log_score(y_true: int, p_hat: float) -> float:
    """
    Binary log-loss per market (single observation).
    """
    p_hat = clip_prob(p_hat)
    # log_loss expects arrays
    return float(log_loss([y_true], [p_hat], labels=[0, 1]))


def summary_features(df_cut: pd.DataFrame) -> Dict[str, float]:
    p = df_cut["price_yes"].to_numpy(dtype=float)
    v = df_cut["volume"].to_numpy(dtype=float)

    if len(p) == 0:
        return {k: np.nan for k in [
            "p_last", "p_ewma", "p_mean", "p_std",
            "v_sum", "v_mean", "n_obs", "p_change", "mean_abs_dp"
        ]}

    dp = np.diff(p) if len(p) > 1 else np.array([0.0])
    return {
        "p_last": float(p[-1]),
        "p_ewma": ewma_last(p, alpha=0.2),
        "p_mean": float(np.mean(p)),
        "p_std": float(np.std(p, ddof=0)),
        "v_sum": float(np.nansum(v)),
        "v_mean": float(np.nanmean(v)) if len(v) else 0.0,
        "n_obs": float(len(p)),
        "p_change": float(p[-1] - p[0]),
        "mean_abs_dp": float(np.mean(np.abs(dp))) if len(dp) else 0.0,
    }


def oof_logreg_probs(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """
    OOF probabilities for logreg baseline (same spirit as Experiment A).
    """
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    y = y.astype(int)

    # Ensure feasible splits
    n_splits = 5
    if len(np.unique(y)) == 2:
        min_class = int(np.bincount(y).min())
        n_splits = min(n_splits, max(2, min_class))
    else:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    p_oof = np.zeros(len(X), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        model = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight="balanced")
        model.fit(X.iloc[train_idx], y[train_idx])
        p_oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    return np.clip(p_oof, EPS, 1.0 - EPS)


# -----------------------------
# Plotting utilities
# -----------------------------
def scatter_plot(x, y, xlabel, ylabel, title, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def threshold_sweep(df: pd.DataFrame, proxy_col: str, score_col: str) -> pd.DataFrame:
    """
    Compute mean score above/below thresholds across a sweep of proxy_col.
    This helps visualize abrupt degradation.
    """
    x = df[proxy_col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return pd.DataFrame()

    lo, hi = np.nanquantile(x, 0.05), np.nanquantile(x, 0.95)
    thresholds = np.linspace(lo, hi, N_THRESHOLD_POINTS)

    rows = []
    for thr in thresholds:
        low = df[df[proxy_col] <= thr]
        high = df[df[proxy_col] > thr]
        if len(low) < MIN_MARKETS_PER_SIDE or len(high) < MIN_MARKETS_PER_SIDE:
            continue
        rows.append({
            "threshold": thr,
            "mean_score_low": float(np.nanmean(low[score_col])),
            "mean_score_high": float(np.nanmean(high[score_col])),
            "n_low": int(len(low)),
            "n_high": int(len(high)),
            "gap_high_minus_low": float(np.nanmean(high[score_col]) - np.nanmean(low[score_col])),
        })
    return pd.DataFrame(rows)


def plot_threshold_sweep(sweep_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if sweep_df.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["threshold"], sweep_df["mean_score_low"], label="Mean log score (proxy <= thr)")
    plt.plot(sweep_df["threshold"], sweep_df["mean_score_high"], label="Mean log score (proxy > thr)")
    plt.xlabel("Threshold on proxy")
    plt.ylabel("Mean log score (lower is better)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_regression_coeffs(coefs: pd.Series, out_path: Path, title: str) -> None:
    coefs = coefs.sort_values(key=lambda s: np.abs(s), ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(coefs)), coefs.to_numpy())
    plt.xticks(np.arange(len(coefs)), coefs.index.tolist(), rotation=45, ha="right")
    plt.ylabel("Coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Main pipeline
# -----------------------------
def main() -> None:
    ensure_dirs()

    outcomes_path = find_outcomes_file()
    outcomes_df = load_outcomes(outcomes_path)
    outcome_map = dict(zip(outcomes_df["name"], outcomes_df["y_true"]))

    rows = []
    feat_rows = []

    # Collect diagnostics and quality at fixed cutoff
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        name = csv_path.stem.strip()
        if name not in outcome_map:
            continue

        df = load_market_csv(csv_path)
        if df.empty or len(df) < 10:
            continue

        y_true = int(outcome_map[name])

        # Cutoff history for "quality"
        df_cut = build_cutoff_history(df, CUTOFF_FRACTION_FOR_QUALITY)

        # Predictions for quality (per-market)
        p_raw = clip_prob(float(df_cut["price_yes"].iloc[-1]))
        p_smooth = clip_prob(ewma_last(df_cut["price_yes"].to_numpy(dtype=float), alpha=0.2))
        p_post = clip_prob(posterior_model_predict(df_cut))

        # Diagnostics (on full series, or you can choose df_cut; we use full for stability)
        sep = separation_proxy(df)
        vol_dx = volatility_dx(df)
        act = activity_volume(df)
        liq = liquidity_proxy(df)

        rows.append({
            "market": name,
            "y_true": y_true,
            "sep_proxy": sep,
            "vol_dx": vol_dx,
            "liq_proxy": liq,
            **act,
            "p_raw": p_raw,
            "p_smoothed": p_smooth,
            "p_posterior": p_post,
            "log_score_raw": market_log_score(y_true, p_raw),
            "log_score_smoothed": market_log_score(y_true, p_smooth),
            "log_score_posterior": market_log_score(y_true, p_post),
        })

        feat_rows.append(summary_features(df_cut))

    if not rows:
        raise RuntimeError("No markets processed. Check data/ and market_outcomes.*")

    diag_df = pd.DataFrame(rows)
    X = pd.DataFrame(feat_rows)
    y = diag_df["y_true"].to_numpy(dtype=int)

    # Optional: compute logreg baseline probs and its log score (for completeness)
    p_logreg = oof_logreg_probs(X, y)
    diag_df["p_logreg"] = p_logreg
    diag_df["log_score_logreg"] = [market_log_score(int(yt), float(ph)) for yt, ph in zip(y, p_logreg)]

    # Save market diagnostics table
    diag_path = REPORTS_DIR / "experiment_C_market_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)

    # Regression: inferential quality (log score) vs diagnostics
    # Choose which quality target to regress: posterior hook (as per paper experiment intent)
    target = "log_score_posterior"
    feature_cols = ["sep_proxy", "vol_dx", "liq_proxy", "v_sum", "v_mean", "v_median", "n_obs", "trades_per_hour"]

    reg_df = diag_df[[target] + feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(reg_df) < 10:
        raise RuntimeError("Not enough finite rows for regression after dropping NaNs.")

    Xr = reg_df[feature_cols].to_numpy(dtype=float)
    yr = reg_df[target].to_numpy(dtype=float)

    lr = LinearRegression()
    lr.fit(Xr, yr)
    coefs = pd.Series(lr.coef_, index=feature_cols)

    reg_summary = pd.DataFrame({
        "feature": feature_cols,
        "coef": lr.coef_,
    }).sort_values("coef", key=lambda s: np.abs(s), ascending=False)

    reg_summary_path = REPORTS_DIR / "experiment_C_regression_summary.csv"
    reg_summary.to_csv(reg_summary_path, index=False)

    # Plots: scatter quality vs key proxies
    scatter_plot(
        x=diag_df["sep_proxy"],
        y=diag_df[target],
        xlabel="Separation proxy (empirical KL-gap proxy)",
        ylabel="Log score (lower is better)",
        title=f"Experiment C — Quality vs separation proxy ({target})",
        out_path=PLOTS_DIR / "experiment_C_scatter_quality_vs_sep_proxy.png",
    )
    scatter_plot(
        x=diag_df["vol_dx"],
        y=diag_df[target],
        xlabel="Volatility of ΔX_t (std of log-odds increments)",
        ylabel="Log score (lower is better)",
        title=f"Experiment C — Quality vs ΔX volatility ({target})",
        out_path=PLOTS_DIR / "experiment_C_scatter_quality_vs_vol_dx.png",
    )

    # Threshold sweep on separation proxy (to highlight abrupt degradation)
    sweep_df = threshold_sweep(diag_df.dropna(subset=["sep_proxy", target]), "sep_proxy", target)
    sweep_path = REPORTS_DIR / "experiment_C_threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    plot_threshold_sweep(
        sweep_df,
        out_path=PLOTS_DIR / "experiment_C_threshold_sweep_sep_proxy.png",
        title=f"Experiment C — Threshold sweep on separation proxy ({target})",
    )

    # Regression coefficient plot
    plot_regression_coeffs(
        coefs=coefs,
        out_path=PLOTS_DIR / "experiment_C_regression_coeffs.png",
        title=f"Experiment C — Linear regression coefficients for {target}",
    )

    # Write report
    report_lines = [
        "# Experiment C — Identifiability Regimes and Failure Diagnostics",
        "",
        "This experiment characterizes regimes where outcome inference becomes ill-posed using market-level diagnostics computed from price–volume histories. We compute (i) a separation proxy (empirical KL-gap proxy), (ii) volatility of log-odds increments ΔX_t, (iii) activity/volume measures, and (iv) a liquidity proxy based on price impact per unit volume. We then regress inferential quality (log score) on these diagnostics and visualize threshold-like degradation in performance.",
        "",
        "## Artifacts",
        "- Market diagnostics: `reports/experiment_C_market_diagnostics.csv`",
        "- Regression summary: `reports/experiment_C_regression_summary.csv`",
        "- Threshold sweep: `reports/experiment_C_threshold_sweep.csv`",
        "- Plots:",
        "  - `plots/experiment_C_scatter_quality_vs_sep_proxy.png`",
        "  - `plots/experiment_C_scatter_quality_vs_vol_dx.png`",
        "  - `plots/experiment_C_threshold_sweep_sep_proxy.png`",
        "  - `plots/experiment_C_regression_coeffs.png`",
        "",
        "## Regression coefficients (absolute-sorted)",
        "",
        reg_summary.to_markdown(index=False) if hasattr(reg_summary, "to_markdown") else reg_summary.to_csv(index=False),
        "",
    ]
    report_path = REPORTS_DIR / "experiment_C_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Done.")
    print(f"- Report:     {report_path}")
    print(f"- Diagnostics:{diag_path}")
    print(f"- Regression: {reg_summary_path}")
    print(f"- Plots:      {PLOTS_DIR.resolve()}")
    print("\nNote: Replace posterior_model_predict(...) with your Bayesian posterior π_t when ready.")


if __name__ == "__main__":
    main()
