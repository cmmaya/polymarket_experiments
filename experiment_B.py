#!/usr/bin/env python3
"""
Experiment B: Posterior Concentration vs. an Empirical KL-Gap Proxy

From experiments.pdf:
- Plot log(1-π_t) (if Y=1) or log(π_t) (if Y=0) as a function of t.
- Compare slopes across bins of an estimated separation proxy (empirical KL-gap proxy).

Assumptions / practical choices (explicit, minimal, runnable):
- We use trade/time index t = 1..T as "t" (same discretization used in Experiment A).
- π_t is produced by the same posterior hook used in Experiment A:
    posterior_model_predict(history_up_to_t)
  For now it defaults to raw price_yes, but you can plug in SMC/AIS/VI later.
- Empirical KL-gap proxy per market: a simple price-based separability proxy that does NOT
  require latent-type parameter inference:
    proxy_i = |mean(logit(P_t)) - logit(P_0)| / (std(diff(logit(P_t))) + eps)
  Intuition: directional drift magnitude relative to increment noise scale.

Outputs (per experiments.pdf):
- plots/experiment_B_concentration_by_bin.png : concentration curves by proxy bin (median + bands)
- plots/experiment_B_slopes_by_bin.png        : slope summaries by proxy bin
- reports/experiment_B_report.md              : intro + bin definitions + slope table

Inputs:
- data/*.csv
- market_outcomes.(csv|tsv|txt) with name,resolved_label
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
N_PROXY_BINS = 4          # compare slopes across bins
N_TIME_GRID = 60          # resample each market to a common grid for cross-market plots
MIN_POINTS_FOR_SLOPE = 15 # minimum grid points to fit slope
ROBUST_Q_LOW, ROBUST_Q_HI = 0.10, 0.90  # for band plots


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
    # Try CSV then TSV
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


def posterior_model_predict(df_cut: pd.DataFrame) -> float:
    """
    Same hook used in Experiment A.
    Replace with your Bayesian posterior inference when ready.
    Placeholder uses last raw price_yes.
    """
    return float(df_cut["price_yes"].iloc[-1])


def empirical_kl_gap_proxy(df: pd.DataFrame) -> float:
    """
    Empirical separation proxy per market (mechanism-agnostic).
    proxy = | mean(X_t) - X_0 | / ( std(ΔX_t) + eps )
    where X_t = logit(P_yes_t)
    """
    p = df["price_yes"].to_numpy(dtype=float)
    if len(p) < 3:
        return float("nan")
    x = logit(p)
    dx = np.diff(x)
    denom = float(np.std(dx, ddof=0) + 1e-8)
    numer = float(abs(np.mean(x) - x[0]))
    return numer / denom


def build_pi_series(df: pd.DataFrame) -> np.ndarray:
    """
    Compute π_t for each t by progressively truncating history.
    Returns array length T with π_1..π_T.
    """
    T = len(df)
    pis = np.zeros(T, dtype=float)
    for t in range(1, T + 1):
        df_cut = df.iloc[:t]
        pis[t - 1] = clip_prob(posterior_model_predict(df_cut))
    return pis


def concentration_target(pi_t: np.ndarray, y_true: int) -> np.ndarray:
    """
    As in experiments.pdf:
      - if Y=1: log(1 - π_t)
      - if Y=0: log(π_t)
    """
    pi_t = np.clip(pi_t, EPS, 1.0 - EPS)
    if y_true == 1:
        return np.log(1.0 - pi_t)
    else:
        return np.log(pi_t)


def resample_to_common_grid(values: np.ndarray, n_grid: int) -> np.ndarray:
    """
    Resample per-market curve to a common length using linear interpolation over normalized time.
    """
    if len(values) == 0:
        return np.full(n_grid, np.nan)
    if len(values) == 1:
        return np.full(n_grid, values[0])

    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, n_grid)
    return np.interp(x_new, x_old, values)


def fit_slope(y: np.ndarray) -> float:
    """
    Fit slope of y(t) over normalized time t in [0,1].
    """
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < MIN_POINTS_FOR_SLOPE:
        return float("nan")
    yv = y[mask].reshape(-1, 1)
    x = np.linspace(0.0, 1.0, len(y)).astype(float)[mask].reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(x, yv)
    return float(lr.coef_.ravel()[0])


# -----------------------------
# Plotting
# -----------------------------
def plot_concentration_by_bin(bin_curves: Dict[str, np.ndarray], out_path: Path) -> None:
    """
    bin_curves: name -> array shape (n_markets_in_bin, n_grid)
    Produces median + [q10,q90] bands per bin.
    """
    plt.figure(figsize=(10, 6))
    for bin_name, curves in bin_curves.items():
        if curves.size == 0:
            continue
        med = np.nanmedian(curves, axis=0)
        ql = np.nanquantile(curves, ROBUST_Q_LOW, axis=0)
        qh = np.nanquantile(curves, ROBUST_Q_HI, axis=0)

        x = np.linspace(0.0, 1.0, curves.shape[1])
        plt.plot(x, med, label=f"{bin_name} (n={curves.shape[0]})")
        plt.fill_between(x, ql, qh, alpha=0.2)

    plt.xlabel("Normalized time (0=start, 1=end)")
    plt.ylabel("Concentration target: log(1-π_t) if Y=1 else log(π_t)")
    plt.title("Experiment B — Posterior concentration vs empirical KL-gap proxy (by bins)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_slopes_by_bin(slopes_df: pd.DataFrame, out_path: Path) -> None:
    """
    Simple bar plot (no manual colors), x=bin, y=median slope, with IQR bars.
    """
    # Prepare stats
    grp = slopes_df.groupby("proxy_bin")["slope"]
    stats = grp.agg(
        n="count",
        median="median",
        q25=lambda s: np.nanquantile(s, 0.25),
        q75=lambda s: np.nanquantile(s, 0.75),
    ).reset_index()

    x = np.arange(len(stats))
    med = stats["median"].to_numpy()
    yerr_low = med - stats["q25"].to_numpy()
    yerr_high = stats["q75"].to_numpy() - med

    plt.figure(figsize=(10, 5))
    plt.bar(x, med, yerr=np.vstack([yerr_low, yerr_high]), capsize=4)
    plt.xticks(x, [f"{b}\n(n={n})" for b, n in zip(stats["proxy_bin"], stats["n"])])
    plt.ylabel("Slope of concentration curve vs normalized time")
    plt.title("Experiment B — Slope summaries by KL-gap proxy bin")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Report
# -----------------------------
def df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def write_report(
    slopes_df: pd.DataFrame,
    proxy_bins_df: pd.DataFrame,
    out_path: Path,
) -> None:
    intro = """# Experiment B — Posterior Concentration vs Empirical KL-Gap Proxy

This experiment tests whether larger empirical separation between outcomes is associated with faster posterior contraction. For each market, we compute a per-market empirical KL-gap proxy and generate a concentration curve defined as log(1-π_t) when the resolved label is YES (Y=1) and log(π_t) when the resolved label is NO (Y=0), where π_t is the inferred probability of YES given the history up to t. We then compare contraction slopes across bins of the empirical proxy.
"""

    # Bin-level summary table
    bin_summary = (
        slopes_df.groupby("proxy_bin")
        .agg(
            n=("slope", "count"),
            slope_median=("slope", "median"),
            slope_q25=("slope", lambda s: np.nanquantile(s, 0.25)),
            slope_q75=("slope", lambda s: np.nanquantile(s, 0.75)),
        )
        .reset_index()
    )

    content = [
        intro,
        "## Proxy definition (empirical KL-gap proxy)",
        df_to_markdown(proxy_bins_df),
        "",
        "## Slope summary by proxy bin",
        df_to_markdown(bin_summary),
        "",
        "## Outputs",
        "- plots/experiment_B_concentration_by_bin.png",
        "- plots/experiment_B_slopes_by_bin.png",
        "",
    ]
    out_path.write_text("\n".join(content), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_dirs()

    outcomes_path = find_outcomes_file()
    outcomes_df = load_outcomes(outcomes_path)
    outcome_map = dict(zip(outcomes_df["name"], outcomes_df["y_true"]))

    # Load markets + compute proxy + concentration curves
    market_rows = []
    curves_resampled = []

    csv_files = sorted(DATA_DIR.glob("*.csv"))
    for csv_path in csv_files:
        name = csv_path.stem.strip()
        if name not in outcome_map:
            continue

        df = load_market_csv(csv_path)
        if df.empty or len(df) < 5:
            continue

        y_true = int(outcome_map[name])

        proxy = empirical_kl_gap_proxy(df)
        if not np.isfinite(proxy):
            continue

        pi_series = build_pi_series(df)
        conc = concentration_target(pi_series, y_true)

        conc_grid = resample_to_common_grid(conc, N_TIME_GRID)

        market_rows.append({"market": name, "y_true": y_true, "proxy": proxy})
        curves_resampled.append(conc_grid)

    if not market_rows:
        raise RuntimeError("No markets were processed (check data/ and market_outcomes).")

    markets_df = pd.DataFrame(market_rows)
    curves_mat = np.vstack(curves_resampled)  # shape (n_markets, n_grid)

    # Bin markets by proxy quantiles
    quantiles = np.linspace(0.0, 1.0, N_PROXY_BINS + 1)
    edges = markets_df["proxy"].quantile(quantiles).to_numpy()
    # Ensure strictly increasing edges for pd.cut
    edges = np.unique(edges)
    if len(edges) < 3:
        # fallback: uniform bins
        edges = np.linspace(markets_df["proxy"].min(), markets_df["proxy"].max(), N_PROXY_BINS + 1)

    # Create bins (labels show interval)
    markets_df["proxy_bin"] = pd.cut(markets_df["proxy"], bins=edges, include_lowest=True)
    markets_df["proxy_bin"] = markets_df["proxy_bin"].astype(str)

    # Fit slope per market (on resampled curve)
    slopes = []
    for i in range(len(markets_df)):
        slope_i = fit_slope(curves_mat[i])
        slopes.append(slope_i)
    markets_df["slope"] = slopes

    slopes_df = markets_df[["market", "proxy", "proxy_bin", "slope", "y_true"]].copy()
    slopes_df.to_csv(REPORTS_DIR / "experiment_B_slopes_by_market.csv", index=False)

    # Prepare curves per bin
    bin_curves: Dict[str, np.ndarray] = {}
    for bin_name, idx in slopes_df.groupby("proxy_bin").groups.items():
        bin_curves[bin_name] = curves_mat[np.array(list(idx)), :]

    # Save plots
    plot_concentration_by_bin(
        bin_curves=bin_curves,
        out_path=PLOTS_DIR / "experiment_B_concentration_by_bin.png",
    )
    plot_slopes_by_bin(
        slopes_df=slopes_df,
        out_path=PLOTS_DIR / "experiment_B_slopes_by_bin.png",
    )

    # Proxy bins description table for report
    proxy_bins_df = pd.DataFrame({
        "proxy_bin": sorted(bin_curves.keys()),
    })

    # Write report
    report_path = REPORTS_DIR / "experiment_B_report.md"
    write_report(slopes_df=slopes_df, proxy_bins_df=proxy_bins_df, out_path=report_path)

    print("Done.")
    print(f"- Report: {report_path}")
    print(f"- Plots:  {PLOTS_DIR.resolve()}")
    print(f"- Slopes: {REPORTS_DIR / 'experiment_B_slopes_by_market.csv'}")
    print("\nNote: Replace posterior_model_predict(...) with your Bayesian posterior π_t when ready.")


if __name__ == "__main__":
    main()
