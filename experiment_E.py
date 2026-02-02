#!/usr/bin/env python3
"""
Experiment E — Information Gain Trajectories

From experiments.pdf (Experiment E) products:
- Plots of mean and quantiles of IG(H_t) vs time-to-resolution and/or cumulative volume
- Identify “information bursts” and relate them to volume spikes
- Report expected information bound via mutual information identity, stratified by design regimes

Given your available data (timestamp, price_yes, volume; outcomes in data/market_outcomes), this script
implements a fully runnable, mechanism-agnostic approximation of IG trajectories:

Definitions used (consistent with binary-outcome Bayesian updating and what you can compute now):
- Prior over outcome: p0 = 0.5 (unless you later provide a different prior).
- Posterior at time t: π_t = posterior_model_predict(H_t)  (hook, placeholder uses last price_yes).
- Instantaneous information gain (posterior vs prior): IG_t = KL(Bern(π_t) || Bern(p0)).
  This equals: π_t log(π_t/p0) + (1-π_t) log((1-π_t)/(1-p0)).
- “Information bursts”: large positive jumps in IG_t (ΔIG_t) above a robust threshold.

Time axis:
- We normalize each market’s timeline to u in [0,1] (0=start of observed data after trimming, 1=end),
  so we can aggregate mean/quantiles across markets.
- We also compute IG as a function of cumulative volume fraction w in [0,1] for the volume-based view.

Design-regime stratification (practical):
- Since “design regimes” are not directly labeled in your repo, we stratify by activity (volume quantiles)
  as a proxy regime. If you later add explicit regime labels, plug them in at `assign_design_regime(...)`.

Outputs:
- plots/experiment_E_IG_vs_time.png
- plots/experiment_E_IG_vs_cumvol.png
- plots/experiment_E_bursts_vs_volume.png
- reports/experiment_E_market_IG_summary.csv
- reports/experiment_E_report.md

Hooks to replace later:
- posterior_model_predict(H_t) -> your Bayesian inverse-problem posterior π_t
- expected information bound via mutual information identity can be added once you define the
  generative likelihood family and prior over latent types; currently we report the generic bound:
  I(Y;H) <= H(Y) with H(Y)=log(2) for p0=0.5, stratified by activity regime.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("data")
PLOTS_DIR = Path("plots")
REPORTS_DIR = Path("reports")

OUTCOMES_PATH_CANDIDATES = [
    DATA_DIR / "market_outcomes",
    DATA_DIR / "market_outcomes.csv",
    DATA_DIR / "market_outcomes.tsv",
    DATA_DIR / "market_outcomes.txt",
    Path("market_outcomes.csv"),
    Path("market_outcomes.tsv"),
    Path("market_outcomes.txt"),
    Path("market_outcomes"),
]

EPS = 1e-6
PRIOR_P0 = 0.5

N_TIME_GRID = 80
N_VOL_GRID = 80

# Burst detection (robust): mark bursts where ΔIG exceeds q90 + k * IQR of ΔIG per market
BURST_Q = 0.90
BURST_IQR_K = 1.5

# Design regime proxy: volume-based bins
N_ACTIVITY_BINS = 3  # low/mid/high by total volume

RNG_SEED = 7


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
        "Could not find outcomes file. Expected one of: "
        + ", ".join(str(p) for p in OUTCOMES_PATH_CANDIDATES)
    )


def load_outcomes(path: Path) -> pd.DataFrame:
    """
    Robust loader for outcomes. Supports CSV/TSV/whitespace.
    Required columns: name, resolved_label
    """
    # Try CSV
    try:
        df = pd.read_csv(path)
        if not {"name", "resolved_label"}.issubset(df.columns):
            raise ValueError()
    except Exception:
        # Try TSV
        try:
            df = pd.read_csv(path, sep="\t")
            if not {"name", "resolved_label"}.issubset(df.columns):
                raise ValueError()
        except Exception:
            # Try whitespace
            df = pd.read_csv(path, sep=r"\s+", engine="python")
            if not {"name", "resolved_label"}.issubset(df.columns):
                raise ValueError(f"Outcomes file must contain columns name,resolved_label. Got {list(df.columns)}")

    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip()
    df["resolved_label"] = df["resolved_label"].astype(str).str.strip().str.upper()
    df = df[df["resolved_label"].isin(["YES", "NO"])].copy()
    df["y_true"] = (df["resolved_label"] == "YES").astype(int)
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
# Core math
# -----------------------------
def clip_prob(p: float) -> float:
    return float(np.clip(p, EPS, 1.0 - EPS))


def posterior_model_predict(df_cut: pd.DataFrame) -> float:
    """
    Hook for π_t := P(Y=1 | H_t). Replace with your Bayesian inverse-problem posterior later.
    Placeholder uses last raw price_yes.
    """
    return float(df_cut["price_yes"].iloc[-1])


def kl_bern(p: float, q: float) -> float:
    """
    KL(Bern(p) || Bern(q))
    """
    p = clip_prob(p)
    q = clip_prob(q)
    return float(p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)))


def build_pi_series(df: pd.DataFrame) -> np.ndarray:
    T = len(df)
    pis = np.zeros(T, dtype=float)
    for t in range(1, T + 1):
        pis[t - 1] = clip_prob(posterior_model_predict(df.iloc[:t]))
    return pis


def resample(values: np.ndarray, n_grid: int) -> np.ndarray:
    if len(values) == 0:
        return np.full(n_grid, np.nan)
    if len(values) == 1:
        return np.full(n_grid, values[0])
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, n_grid)
    return np.interp(x_new, x_old, values)


def resample_vs_cumvol(values: np.ndarray, volume: np.ndarray, n_grid: int) -> np.ndarray:
    """
    Resample values as a function of cumulative volume fraction in [0,1].
    """
    if len(values) == 0:
        return np.full(n_grid, np.nan)
    if len(values) == 1:
        return np.full(n_grid, values[0])

    v = np.asarray(volume, dtype=float)
    v = np.maximum(0.0, v)
    cum = np.cumsum(v)
    total = float(cum[-1]) if len(cum) else 0.0
    if total <= 0:
        # fallback to time resample
        return resample(values, n_grid)

    w_old = cum / total
    w_old = np.clip(w_old, 0.0, 1.0)

    w_new = np.linspace(0.0, 1.0, n_grid)
    # Ensure monotonic interpolation
    # If w_old has flat segments, np.interp still works but may collapse.
    return np.interp(w_new, w_old, values)


def detect_bursts(ig: np.ndarray) -> np.ndarray:
    """
    Returns boolean mask for burst indices (on ΔIG, aligned to t>=2).
    """
    if len(ig) < 3:
        return np.zeros(len(ig), dtype=bool)
    dig = np.diff(ig, prepend=ig[0])
    # robust threshold: q90 + k*IQR of ΔIG (per market)
    q90 = float(np.nanquantile(dig, BURST_Q))
    q25 = float(np.nanquantile(dig, 0.25))
    q75 = float(np.nanquantile(dig, 0.75))
    thr = q90 + BURST_IQR_K * (q75 - q25)
    return dig > thr


def assign_design_regime(total_volume: float, edges: np.ndarray) -> str:
    # edges length = bins+1
    for i in range(len(edges) - 1):
        if total_volume >= edges[i] and total_volume <= edges[i + 1] + 1e-12:
            return f"activity_bin_{i+1}"
    return "activity_bin_unknown"


# -----------------------------
# Plotting
# -----------------------------
def plot_mean_quantiles(curves: np.ndarray, x: np.ndarray, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    """
    curves shape: (n_markets, n_grid)
    """
    med = np.nanmedian(curves, axis=0)
    q10 = np.nanquantile(curves, 0.10, axis=0)
    q90 = np.nanquantile(curves, 0.90, axis=0)
    mean = np.nanmean(curves, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(x, mean, label="mean")
    plt.plot(x, med, label="median")
    plt.fill_between(x, q10, q90, alpha=0.2, label="q10-q90")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_bursts_vs_volume(burst_rows: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter: burst magnitude vs concurrent volume and mark activity regime.
    """
    if burst_rows.empty:
        return
    plt.figure(figsize=(9, 6))
    plt.scatter(burst_rows["volume"], burst_rows["delta_ig"], alpha=0.8)
    plt.xlabel("Volume at burst time (takerOnly)")
    plt.ylabel("ΔIG at burst time")
    plt.title("Experiment E — Information bursts vs volume (all markets)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_dirs()

    outcomes_path = find_outcomes_file()
    outcomes_df = load_outcomes(outcomes_path)
    outcome_map = dict(zip(outcomes_df["name"], outcomes_df["y_true"]))

    market_summaries = []
    ig_time_curves = []
    ig_vol_curves = []

    burst_records = []

    # First pass: compute per-market IG, store total volumes for regime bins
    total_vols = []

    per_market_cache: Dict[str, Dict] = {}

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        name = csv_path.stem.strip()
        if name not in outcome_map:
            continue

        df = load_market_csv(csv_path)
        if df.empty or len(df) < 5:
            continue

        pis = build_pi_series(df)
        ig = np.array([kl_bern(float(pi), PRIOR_P0) for pi in pis], dtype=float)

        vol = df["volume"].to_numpy(dtype=float)
        vol = np.maximum(0.0, vol)
        total_vol = float(np.nansum(vol))

        total_vols.append(total_vol)
        per_market_cache[name] = {
            "ig": ig,
            "vol": vol,
            "total_vol": total_vol,
        }

    if not per_market_cache:
        raise RuntimeError("No markets processed. Check data/*.csv and outcomes file.")

    # Activity-bin edges (design regimes proxy)
    total_vols_arr = np.asarray(total_vols, dtype=float)
    edges = np.nanquantile(total_vols_arr, [0.0, 1/3, 2/3, 1.0])
    edges = np.unique(edges)
    if len(edges) < 4:
        edges = np.linspace(np.nanmin(total_vols_arr), np.nanmax(total_vols_arr), N_ACTIVITY_BINS + 1)

    # Second pass: resample curves + bursts
    for name, obj in per_market_cache.items():
        ig = obj["ig"]
        vol = obj["vol"]
        total_vol = float(obj["total_vol"])
        regime = assign_design_regime(total_vol, edges)

        ig_time = resample(ig, N_TIME_GRID)
        ig_vol = resample_vs_cumvol(ig, vol, N_VOL_GRID)

        ig_time_curves.append(ig_time)
        ig_vol_curves.append(ig_vol)

        # bursts
        burst_mask = detect_bursts(ig)
        dig = np.diff(ig, prepend=ig[0])
        idxs = np.where(burst_mask)[0]
        for t in idxs:
            burst_records.append({
                "market": name,
                "t_index": int(t),
                "delta_ig": float(dig[t]),
                "volume": float(vol[t]) if t < len(vol) else float("nan"),
                "regime": regime,
            })

        # per-market summary
        market_summaries.append({
            "market": name,
            "regime": regime,
            "n_obs": int(len(ig)),
            "total_volume": total_vol,
            "IG_final": float(ig[-1]),
            "IG_max": float(np.nanmax(ig)),
            "n_bursts": int(len(idxs)),
        })

    ig_time_mat = np.vstack(ig_time_curves)
    ig_vol_mat = np.vstack(ig_vol_curves)

    # Save summary CSVs
    summary_df = pd.DataFrame(market_summaries).sort_values(["regime", "IG_final"], ascending=[True, False])
    summary_path = REPORTS_DIR / "experiment_E_market_IG_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    bursts_df = pd.DataFrame(burst_records)
    bursts_path = REPORTS_DIR / "experiment_E_bursts.csv"
    bursts_df.to_csv(bursts_path, index=False)

    # Plots
    x_time = np.linspace(0.0, 1.0, N_TIME_GRID)
    plot_mean_quantiles(
        curves=ig_time_mat,
        x=x_time,
        title="Experiment E — IG(H_t) vs normalized time",
        xlabel="Normalized time (0=start, 1=end)",
        ylabel="IG(H_t) = KL(Bern(π_t) || Bern(prior))",
        out_path=PLOTS_DIR / "experiment_E_IG_vs_time.png",
    )

    x_vol = np.linspace(0.0, 1.0, N_VOL_GRID)
    plot_mean_quantiles(
        curves=ig_vol_mat,
        x=x_vol,
        title="Experiment E — IG(H_t) vs cumulative volume fraction",
        xlabel="Cumulative volume fraction",
        ylabel="IG(H_t) = KL(Bern(π_t) || Bern(prior))",
        out_path=PLOTS_DIR / "experiment_E_IG_vs_cumvol.png",
    )

    plot_bursts_vs_volume(
        burst_rows=bursts_df,
        out_path=PLOTS_DIR / "experiment_E_bursts_vs_volume.png",
    )

    # Information bound (generic) and stratification
    # With prior p0, outcome entropy H(Y)= -p0 log p0 - (1-p0) log(1-p0)
    HY = float(-(PRIOR_P0 * math.log(PRIOR_P0) + (1 - PRIOR_P0) * math.log(1 - PRIOR_P0)))
    # Report empirical final IG by regime vs HY
    bound_df = (
        summary_df.groupby("regime")
        .agg(
            n=("market", "count"),
            IG_final_mean=("IG_final", "mean"),
            IG_final_median=("IG_final", "median"),
            IG_final_q90=("IG_final", lambda s: float(np.nanquantile(s, 0.90))),
            total_volume_median=("total_volume", "median"),
        )
        .reset_index()
    )
    bound_df["H(Y)_bound"] = HY

    bound_path = REPORTS_DIR / "experiment_E_bound_by_regime.csv"
    bound_df.to_csv(bound_path, index=False)

    # Report (md)
    def to_md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_csv(index=False)

    report_lines = [
        "# Experiment E — Information Gain Trajectories",
        "",
        "This experiment analyzes how information about the binary outcome accumulates over time in market histories. "
        "For each market we compute a posterior trajectory π_t from the observed history H_t and define the information gain "
        "trajectory as IG(H_t) = KL(Bern(π_t) || Bern(prior)), using a symmetric prior p0=0.5. We aggregate trajectories across markets "
        "as a function of normalized time and cumulative volume fraction, and identify information bursts using robust thresholds on ΔIG.",
        "",
        "## Artifacts",
        "- `plots/experiment_E_IG_vs_time.png`",
        "- `plots/experiment_E_IG_vs_cumvol.png`",
        "- `plots/experiment_E_bursts_vs_volume.png`",
        "- `reports/experiment_E_market_IG_summary.csv`",
        "- `reports/experiment_E_bursts.csv`",
        "- `reports/experiment_E_bound_by_regime.csv`",
        "",
        "## Expected information bound (generic)",
        f"- Outcome entropy bound: H(Y) = {HY:.6f} (nats) for prior p0={PRIOR_P0:.2f}.",
        "",
        "## Empirical IG_final by activity regime (proxy design regimes)",
        "",
        to_md(bound_df),
        "",
        "## Notes",
        "- Replace posterior_model_predict(H_t) with your Bayesian inverse-problem posterior π_t to match the paper model.",
        "- If you later define explicit market-design regimes, replace the activity-bin regime assignment with those labels.",
        "",
    ]
    report_path = REPORTS_DIR / "experiment_E_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Done.")
    print(f"- Report:   {report_path}")
    print(f"- Summary:  {summary_path}")
    print(f"- Bursts:   {bursts_path}")
    print(f"- Bound:    {bound_path}")
    print(f"- Plots:    {PLOTS_DIR.resolve()}")
    print("\nNote: Replace posterior_model_predict(...) with your Bayesian posterior π_t when ready.")


if __name__ == "__main__":
    main()
