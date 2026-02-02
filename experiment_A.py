#!/usr/bin/env python3
"""
Experiment A: Backtest Outcome Inference Across Resolved Markets (Polymarket-style CSVs)

Inputs (expected):
- data/*.csv : one file per market. Filename stem is market name, e.g. "culture-gta-vi-released-in-2025"
  Required columns: timestamp, price_yes, price_no, volume
  Notes: volume is takerOnly (fine). price_yes is treated as market-implied probability P_t of YES.

- market_outcomes.(csv|tsv|txt) : outcome labels with columns:
    name,resolved_label
  where resolved_label in {"YES","NO"}.

Outputs:
- plots/*.png : reliability diagrams + metric summaries (per cutoff, per model).
- reports/experiment_A_report.md : report with intro + metrics table.

Baselines (as in experiments.pdf):
- raw market price P_t
- smoothed price
- logistic regression on summary features

Posterior π_t hook:
- Replace posterior_model_predict(...) with your Bayesian inverse-problem inference when ready.
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt


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

# Cutoffs as in experiments.pdf (time-to-close fractions / trade-count quantiles).
# With only history CSV, we implement "fraction of samples in the series".
CUTOFF_FRACTIONS = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

N_ECE_BINS = 10
EPS = 1e-6  # stability for clipping probs


# -----------------------------
# Helpers: loading
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
        # attempt comma first, fallback to tab
        try:
            df = pd.read_csv(path)
            # heuristic: if single column and tab present, re-read as TSV
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

    # Parse timestamp (robust)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Numeric columns
    for c in ["price_yes", "price_no", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Cut rows until first non-null price_yes
    first_valid = df["price_yes"].first_valid_index()
    if first_valid is None:
        return df.iloc[0:0].copy()
    df = df.loc[first_valid:].reset_index(drop=True)

    # Drop rows where price_yes is missing
    df = df.dropna(subset=["price_yes"]).reset_index(drop=True)

    return df


# -----------------------------
# Features / predictors
# -----------------------------
def clip_prob(p: float) -> float:
    return float(np.clip(p, EPS, 1.0 - EPS))


def ewma_last(prices: np.ndarray, alpha: float = 0.2) -> float:
    if len(prices) == 0:
        return float("nan")
    s = float(prices[0])
    for x in prices[1:]:
        s = alpha * float(x) + (1 - alpha) * s
    return float(s)


def summary_features(df_cut: pd.DataFrame) -> Dict[str, float]:
    p = df_cut["price_yes"].to_numpy(dtype=float)
    v = df_cut["volume"].to_numpy(dtype=float)

    if len(p) == 0:
        return {
            "p_last": np.nan,
            "p_ewma": np.nan,
            "p_mean": np.nan,
            "p_std": np.nan,
            "v_sum": np.nan,
            "v_mean": np.nan,
            "n_obs": np.nan,
            "p_change": np.nan,
            "mean_abs_dp": np.nan,
        }

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


def posterior_model_predict(df_cut: pd.DataFrame) -> float:
    """
    Hook for π_t := P(Y=1 | H_t). Replace with SMC/AIS/VI inference later.
    Placeholder keeps pipeline runnable: returns last raw price_yes.
    """
    return float(df_cut["price_yes"].iloc[-1])


# -----------------------------
# Metrics: ECE + reliability
# -----------------------------
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_prob[mask]))
        w = float(np.mean(mask))
        ece += w * abs(acc - conf)
    return float(ece)


def reliability_curve_points(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confs, accs = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        confs.append(float(np.mean(y_prob[mask])))
        accs.append(float(np.mean(y_true[mask])))
    return np.array(confs, dtype=float), np.array(accs, dtype=float)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, EPS, 1.0 - EPS)

    out = {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_score": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=N_ECE_BINS),
    }
    if len(np.unique(y_true)) == 2:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["auc"] = float("nan")
    return out


# -----------------------------
# Plotting
# -----------------------------
def save_reliability_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    out_path: Path,
    n_bins: int = 10,
) -> None:
    confs, accs = reliability_curve_points(y_true, y_prob, n_bins=n_bins)

    plt.figure()
    # Perfect calibration line
    plt.plot([0, 1], [0, 1])
    # Reliability points
    plt.plot(confs, accs, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_metric_table_plot(metrics_df: pd.DataFrame, out_path: Path, title: str) -> None:
    """
    Simple visualization: save metrics_df as an image table using matplotlib.
    """
    plt.figure(figsize=(12, max(2.5, 0.35 * (len(metrics_df) + 1))))
    plt.axis("off")
    tbl = plt.table(
        cellText=metrics_df.round(4).values,
        colLabels=metrics_df.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Experiment A pipeline
# -----------------------------
@dataclass
class CutoffResult:
    market_name: str
    cutoff_fraction: float
    y_true: int
    p_raw: float
    p_smoothed: float
    p_posterior: float  # hook output
    p_logreg: Optional[float] = None


def build_cutoff_history(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if df.empty:
        return df
    n = len(df)
    k = max(1, int(math.floor(frac * n)))
    return df.iloc[:k].copy()


def run_experiment_A(
    data_dir: Path,
    outcomes_df: pd.DataFrame,
    cutoff_fracs: List[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outcome_map = dict(zip(outcomes_df["name"], outcomes_df["y_true"]))

    rows: List[CutoffResult] = []
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    # Build per-market predictions at each cutoff
    for csv_path in csv_files:
        market_name = csv_path.stem.strip()
        if market_name not in outcome_map:
            continue

        df = load_market_csv(csv_path)
        if df.empty:
            continue

        y_true = int(outcome_map[market_name])

        for frac in cutoff_fracs:
            df_cut = build_cutoff_history(df, frac)
            if df_cut.empty:
                continue

            p_raw = clip_prob(float(df_cut["price_yes"].iloc[-1]))
            p_smooth = clip_prob(ewma_last(df_cut["price_yes"].to_numpy(dtype=float), alpha=0.2))
            p_post = clip_prob(posterior_model_predict(df_cut))

            rows.append(
                CutoffResult(
                    market_name=market_name,
                    cutoff_fraction= frac,
                    y_true=y_true,
                    p_raw=p_raw,
                    p_smoothed=p_smooth,
                    p_posterior=p_post,
                )
            )

    if not rows:
        raise RuntimeError("No matching markets found between data/*.csv and market_outcomes.*")

    per_df = pd.DataFrame([r.__dict__ for r in rows])

    # Logistic regression baseline
    feats_list = []
    for r in rows:
        df_full = load_market_csv(data_dir / f"{r.market_name}.csv")
        df_cut = build_cutoff_history(df_full, r.cutoff_fraction)
        feats_list.append(summary_features(df_cut))

    X = pd.DataFrame(feats_list)
    y = per_df["y_true"].astype(int).to_numpy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Out-of-fold prediction
    # (Note: strict grouping by market can be added later with GroupKFold.)
    n_splits = 5
    if len(np.unique(y)) == 2:
        # ensure enough samples per class
        min_class = int(np.bincount(y).min())
        n_splits = min(n_splits, max(2, min_class))
    else:
        n_splits = 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
    p_oof = np.zeros(len(X), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        p_oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    per_df["p_logreg"] = np.clip(p_oof, EPS, 1.0 - EPS)

    # Aggregate metrics by cutoff and model
    metric_rows = []
    for frac in cutoff_fracs:
        sub = per_df[per_df["cutoff_fraction"] == frac]
        if sub.empty:
            continue

        y_true = sub["y_true"].to_numpy()
        for model_name, col in [
            ("raw_price", "p_raw"),
            ("smoothed_price", "p_smoothed"),
            ("posterior_hook", "p_posterior"),
            ("logreg", "p_logreg"),
        ]:
            y_prob = sub[col].to_numpy(dtype=float)
            m = compute_metrics(y_true, y_prob)
            metric_rows.append({
                "cutoff_fraction": frac,
                "model": model_name,
                "brier": m["brier"],
                "log_score": m["log_score"],
                "auc": m["auc"],
                "ece": m["ece"],
                "n_samples": int(len(sub)),
            })

    metrics_df = (
        pd.DataFrame(metric_rows)
        .sort_values(["cutoff_fraction", "model"])
        .reset_index(drop=True)
    )

    return per_df, metrics_df


# -----------------------------
# Reporting
# -----------------------------
def df_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        # fallback (no tabulate)
        return df.to_csv(index=False)


def write_report(
    per_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    out_path: Path,
) -> None:
    # Provide a concise technical intro (as requested)
    intro = """# Experiment A — Backtest Outcome Inference Across Resolved Markets

This experiment evaluates outcome inference quality across a universe of resolved binary markets. For each market, we construct truncated histories at multiple cutoffs (fractions of the available history length) and compute predicted probabilities of the resolved label using: (i) the raw market price (YES probability), (ii) a smoothed price baseline, (iii) a simple logistic regression trained on summary features, and (iv) a placeholder hook for the Bayesian posterior π_t (to be replaced by the inverse-problem inference model). Predictions are scored against the audited resolution labels using Brier score, log score, AUC, and calibration metrics (reliability diagrams and ECE).
"""

    # Metrics pivot for readability
    pivot = metrics_df.pivot_table(
        index=["cutoff_fraction", "model"],
        values=["brier", "log_score", "auc", "ece", "n_samples"],
        aggfunc="first",
    ).reset_index()

    # Also include overall pooled metrics per model (all cutoffs pooled)
    pooled_rows = []
    for model_name, col in [
        ("raw_price", "p_raw"),
        ("smoothed_price", "p_smoothed"),
        ("posterior_hook", "p_posterior"),
        ("logreg", "p_logreg"),
    ]:
        m = compute_metrics(per_df["y_true"].to_numpy(), per_df[col].to_numpy(dtype=float))
        pooled_rows.append({
            "model": model_name,
            "brier": m["brier"],
            "log_score": m["log_score"],
            "auc": m["auc"],
            "ece": m["ece"],
            "n_samples": int(len(per_df)),
        })
    pooled_df = pd.DataFrame(pooled_rows).sort_values("model").reset_index(drop=True)

    content = [
        intro,
        "## Aggregated metrics by cutoff and model",
        df_to_markdown(pivot),
        "",
        "## Pooled metrics across all cutoffs",
        df_to_markdown(pooled_df),
        "",
        "## Notes on artifacts",
        "- Reliability diagrams are saved under `plots/` for each cutoff and model.",
        "- A tabular summary of metrics is also saved as an image under `plots/`.",
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

    per_df, metrics_df = run_experiment_A(DATA_DIR, outcomes_df, CUTOFF_FRACTIONS)

    # Save metric table as image
    save_metric_table_plot(
        metrics_df=metrics_df,
        out_path=PLOTS_DIR / "experiment_A_metrics_table.png",
        title="Experiment A — Metrics by cutoff and model",
    )

    # Reliability plots per cutoff & model
    for frac in sorted(per_df["cutoff_fraction"].unique()):
        sub = per_df[per_df["cutoff_fraction"] == frac]
        y_true = sub["y_true"].to_numpy()

        for model_name, col in [
            ("raw_price", "p_raw"),
            ("smoothed_price", "p_smoothed"),
            ("posterior_hook", "p_posterior"),
            ("logreg", "p_logreg"),
        ]:
            y_prob = sub[col].to_numpy(dtype=float)
            title = f"Reliability — {model_name} — cutoff={frac:.2f} (n={len(sub)})"
            out_path = PLOTS_DIR / f"experiment_A_reliability_{model_name}_cutoff_{frac:.2f}.png"
            save_reliability_plot(y_true, y_prob, title=title, out_path=out_path, n_bins=N_ECE_BINS)

    # Write markdown report
    report_path = REPORTS_DIR / "experiment_A_report.md"
    write_report(per_df, metrics_df, report_path)

    # Also save raw outputs for inspection
    per_df.to_csv(REPORTS_DIR / "experiment_A_predictions_by_market_cutoff.csv", index=False)
    metrics_df.to_csv(REPORTS_DIR / "experiment_A_metrics.csv", index=False)

    print("Done.")
    print(f"- Report:  {report_path}")
    print(f"- Plots:   {PLOTS_DIR.resolve()}")
    print(f"- Tables:  {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
