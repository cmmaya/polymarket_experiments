#!/usr/bin/env python3
"""
Experiment D — Stability Under Realistic Perturbations

From experiments.pdf (Experiment D) products:
- Distributions of |log BF(H) - log BF(H')|
- Distributions of |π(H) - π(H')|
- Stratified by (i) activity level and (ii) truncation event E_R

Repo conventions (your setup):
- data/*.csv : one per market, filename stem is market name (category-prefixed)
  columns: timestamp, price_yes, price_no, volume  (volume = takerOnly)
- outcomes file: data/market_outcomes (or .csv/.tsv/.txt)
  format can be whitespace-separated or CSV/TSV with columns: name, resolved_label

Implementation choices (runnable, minimal):
- H is the truncated history at cutoff_fraction (default 0.90).
- H' is the perturbed version of the same truncated history.
- π(H) is computed via posterior_model_predict(H) (hook). Default uses last price_yes.
- log BF(H) is computed from π using log-odds (prior odds = 1): logBF := logit(π).
  This lets you produce the required Bayes-factor-difference artifact without needing an
  explicit BF from the inference engine. If later your model outputs BF directly,
  replace logBF_from_pi(...) accordingly.
- Perturbations implemented:
    (1) price rounding (to 3 decimals)                -> H'
    (2) downsampling (keep every k-th row)            -> H'
    (3) additive noise in log-odds X=logit(P)         -> H'
    (4) volume perturbation (mult. noise on volume)   -> H'  (affects only if your posterior uses volume)

Outputs:
- plots/experiment_D_hist_logBF_<perturb>_<stratum>.png
- plots/experiment_D_hist_pi_<perturb>_<stratum>.png
- reports/experiment_D_deltas.csv
- reports/experiment_D_summary.csv
- reports/experiment_D_report.md
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

# History truncation for Experiment D
CUTOFF_FRACTION = 0.90

# Activity stratification (by total takerOnly volume in truncated history)
N_ACTIVITY_BINS = 3  # low / mid / high by quantiles

# Perturbation settings
ROUND_DECIMALS = 3

DOWNSAMPLE_KS = [2, 5, 10]           # keep every k-th row
LOGODDS_NOISE_SIGMAS = [0.05, 0.10]  # gaussian noise std on X=logit(P)
VOLUME_MULT_SIGMAS = [0.10, 0.25]    # multiplicative noise on volume

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
    Robust loader for outcomes. Supports:
    - CSV (comma), TSV (tab), or whitespace-separated (as shown in your example).
    Required columns: name, resolved_label
    """
    text_head = path.read_text(encoding="utf-8", errors="ignore")[:2000]

    # Try CSV
    try:
        df = pd.read_csv(path)
        if {"name", "resolved_label"}.issubset(df.columns):
            pass
        else:
            raise ValueError("Not a standard CSV outcomes file.")
    except Exception:
        # Try TSV
        try:
            df = pd.read_csv(path, sep="\t")
            if not {"name", "resolved_label"}.issubset(df.columns):
                raise ValueError("Not a TSV outcomes file.")
        except Exception:
            # Try whitespace-separated
            df = pd.read_csv(path, sep=r"\s+", engine="python")
            if not {"name", "resolved_label"}.issubset(df.columns):
                raise ValueError(f"Outcomes file must contain columns name,resolved_label. Got {list(df.columns)}")

    df = df.copy()
    df["name"] = df["name"].astype(str).str.strip()
    df["resolved_label"] = df["resolved_label"].astype(str).str.strip().str.upper()
    df = df[df["resolved_label"].isin(["YES", "NO"])].copy()
    df["y_true"] = (df["resolved_label"] == "YES").astype(int)
    return df[["name", "resolved_label", "y_true"]]


def load_market_csv(csv_path: Path) -> Tuple[pd.DataFrame, bool]:
    """
    Returns (df, E_R_flag) where E_R_flag captures the truncation event:
    E_R = True if we had to drop an initial block due to leading null price_yes
          (i.e., first_valid_index > 0).
    """
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
        return df.iloc[0:0].copy(), False

    E_R = bool(first_valid > 0)
    df = df.loc[first_valid:].reset_index(drop=True)
    df = df.dropna(subset=["price_yes"]).reset_index(drop=True)

    return df, E_R


# -----------------------------
# Core math
# -----------------------------
def clip_prob(p: float) -> float:
    return float(np.clip(p, EPS, 1.0 - EPS))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def inv_logit(x: np.ndarray) -> np.ndarray:
    # stable logistic
    x = x.astype(float)
    return 1.0 / (1.0 + np.exp(-x))


def build_cutoff_history(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    if df.empty:
        return df
    n = len(df)
    k = max(1, int(math.floor(frac * n)))
    return df.iloc[:k].copy()


def posterior_model_predict(df_cut: pd.DataFrame) -> float:
    """
    Hook for π(H). Replace with your Bayesian inverse-problem posterior when ready.
    Placeholder: last raw price_yes.
    """
    return float(df_cut["price_yes"].iloc[-1])


def logBF_from_pi(pi: float) -> float:
    """
    log BF(H) computed from posterior probability π(H) under prior odds = 1:
      BF = odds(π) / odds(prior) = odds(π) since odds(prior)=1 if prior=0.5
      logBF = logit(π)
    """
    pi = clip_prob(pi)
    return float(math.log(pi / (1.0 - pi)))


# -----------------------------
# Perturbations (produce H')
# -----------------------------
def perturb_round_prices(df: pd.DataFrame, decimals: int) -> pd.DataFrame:
    out = df.copy()
    out["price_yes"] = out["price_yes"].round(decimals)
    # keep within [eps, 1-eps]
    out["price_yes"] = out["price_yes"].clip(EPS, 1.0 - EPS)
    return out


def perturb_downsample(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 1 or df.empty:
        return df.copy()
    out = df.iloc[::k].copy()
    if len(out) == 0:
        return df.iloc[:1].copy()
    return out.reset_index(drop=True)


def perturb_logodds_noise(df: pd.DataFrame, sigma: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    p = out["price_yes"].to_numpy(dtype=float)
    x = logit(p)
    noise = rng.normal(loc=0.0, scale=sigma, size=len(x))
    x2 = x + noise
    p2 = inv_logit(x2)
    out["price_yes"] = np.clip(p2, EPS, 1.0 - EPS)
    return out


def perturb_volume_mult(df: pd.DataFrame, sigma: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    v = out["volume"].to_numpy(dtype=float)
    mult = rng.normal(loc=1.0, scale=sigma, size=len(v))
    v2 = v * mult
    # volume can't be negative
    out["volume"] = np.maximum(0.0, v2)
    return out


# -----------------------------
# Plotting
# -----------------------------
def save_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path) -> None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Experiment D runner
# -----------------------------
@dataclass
class DeltaRow:
    market: str
    perturbation: str
    activity_bin: str
    E_R: bool
    v_sum: float
    n_obs: int
    delta_pi: float
    delta_logBF: float


def main() -> None:
    ensure_dirs()
    rng = np.random.default_rng(RNG_SEED)

    outcomes_path = find_outcomes_file()
    outcomes_df = load_outcomes(outcomes_path)
    outcome_map = dict(zip(outcomes_df["name"], outcomes_df["y_true"]))

    market_records: List[Dict] = []
    deltas: List[DeltaRow] = []

    # First pass: load markets, compute baseline π(H), activity stats, E_R, etc.
    markets_data: Dict[str, Dict] = {}

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        name = csv_path.stem.strip()
        if name not in outcome_map:
            continue

        df, E_R = load_market_csv(csv_path)
        if df.empty or len(df) < 5:
            continue

        df_H = build_cutoff_history(df, CUTOFF_FRACTION)

        # baseline π(H), logBF(H)
        pi_H = clip_prob(posterior_model_predict(df_H))
        logBF_H = logBF_from_pi(pi_H)

        v_sum = float(np.nansum(df_H["volume"].to_numpy(dtype=float)))
        n_obs = int(len(df_H))

        markets_data[name] = {
            "df_H": df_H,
            "pi_H": pi_H,
            "logBF_H": logBF_H,
            "v_sum": v_sum,
            "n_obs": n_obs,
            "E_R": E_R,
        }

        market_records.append({
            "market": name,
            "v_sum": v_sum,
            "n_obs": n_obs,
            "E_R": E_R,
        })

    if not markets_data:
        raise RuntimeError("No markets processed. Check data/*.csv and outcomes file.")

    markets_df = pd.DataFrame(market_records)

    # Activity bins by v_sum quantiles
    qs = markets_df["v_sum"].quantile([0.0, 1/3, 2/3, 1.0]).to_numpy()
    # Ensure edges strictly increasing
    qs = np.unique(qs)
    if len(qs) < 4:
        qs = np.linspace(markets_df["v_sum"].min(), markets_df["v_sum"].max(), N_ACTIVITY_BINS + 1)

    markets_df["activity_bin"] = pd.cut(markets_df["v_sum"], bins=qs, include_lowest=True).astype(str)
    activity_bin_map = dict(zip(markets_df["market"], markets_df["activity_bin"]))

    # Define perturbation list
    perturb_specs: List[Tuple[str, Dict]] = []
    perturb_specs.append((f"round_decimals_{ROUND_DECIMALS}", {"type": "round"}))
    for k in DOWNSAMPLE_KS:
        perturb_specs.append((f"downsample_k_{k}", {"type": "downsample", "k": k}))
    for s in LOGODDS_NOISE_SIGMAS:
        perturb_specs.append((f"logodds_noise_sigma_{s}", {"type": "logodds_noise", "sigma": s}))
    for s in VOLUME_MULT_SIGMAS:
        perturb_specs.append((f"volume_mult_sigma_{s}", {"type": "volume_mult", "sigma": s}))

    # Apply perturbations and record deltas
    for market, info in markets_data.items():
        df_H = info["df_H"]
        pi_H = float(info["pi_H"])
        logBF_H = float(info["logBF_H"])
        v_sum = float(info["v_sum"])
        n_obs = int(info["n_obs"])
        E_R = bool(info["E_R"])
        activity_bin = activity_bin_map.get(market, "unknown")

        for pname, spec in perturb_specs:
            if spec["type"] == "round":
                df_Hp = perturb_round_prices(df_H, ROUND_DECIMALS)
            elif spec["type"] == "downsample":
                df_Hp = perturb_downsample(df_H, int(spec["k"]))
            elif spec["type"] == "logodds_noise":
                df_Hp = perturb_logodds_noise(df_H, float(spec["sigma"]), rng)
            elif spec["type"] == "volume_mult":
                df_Hp = perturb_volume_mult(df_H, float(spec["sigma"]), rng)
            else:
                continue

            pi_Hp = clip_prob(posterior_model_predict(df_Hp))
            logBF_Hp = logBF_from_pi(pi_Hp)

            deltas.append(
                DeltaRow(
                    market=market,
                    perturbation=pname,
                    activity_bin=activity_bin,
                    E_R=E_R,
                    v_sum=v_sum,
                    n_obs=n_obs,
                    delta_pi=float(abs(pi_H - pi_Hp)),
                    delta_logBF=float(abs(logBF_H - logBF_Hp)),
                )
            )

    deltas_df = pd.DataFrame([d.__dict__ for d in deltas])
    deltas_path = REPORTS_DIR / "experiment_D_deltas.csv"
    deltas_df.to_csv(deltas_path, index=False)

    # Summaries (overall + stratified)
    summary = (
        deltas_df.groupby(["perturbation", "activity_bin", "E_R"])
        .agg(
            n=("delta_pi", "count"),
            delta_pi_median=("delta_pi", "median"),
            delta_pi_q90=("delta_pi", lambda s: float(np.nanquantile(s, 0.90))),
            delta_logBF_median=("delta_logBF", "median"),
            delta_logBF_q90=("delta_logBF", lambda s: float(np.nanquantile(s, 0.90))),
        )
        .reset_index()
        .sort_values(["perturbation", "activity_bin", "E_R"])
    )
    summary_path = REPORTS_DIR / "experiment_D_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Plots: histograms per (perturbation, stratum)
    # Stratum = activity_bin x E_R
    for (pert, act_bin, E_R_flag), sub in deltas_df.groupby(["perturbation", "activity_bin", "E_R"]):
        tag = f"{pert}__act_{act_bin.replace(' ', '').replace(',', '_').replace('[','').replace(']','').replace('(','').replace(')','')}"
        tag += f"__ER_{int(E_R_flag)}"

        save_hist(
            sub["delta_logBF"].to_numpy(dtype=float),
            title=f"|logBF(H)-logBF(H')| — {pert} — activity={act_bin} — E_R={E_R_flag} — n={len(sub)}",
            xlabel="|Δ logBF|",
            out_path=PLOTS_DIR / f"experiment_D_hist_logBF_{tag}.png",
        )
        save_hist(
            sub["delta_pi"].to_numpy(dtype=float),
            title=f"|π(H)-π(H')| — {pert} — activity={act_bin} — E_R={E_R_flag} — n={len(sub)}",
            xlabel="|Δ π|",
            out_path=PLOTS_DIR / f"experiment_D_hist_pi_{tag}.png",
        )

    # Markdown report (concise)
    def to_md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_csv(index=False)

    report_lines = [
        "# Experiment D — Stability Under Realistic Perturbations",
        "",
        "This experiment evaluates stability of inference under realistic perturbations of the observed history. "
        f"For each market we define H as the truncated history at cutoff_fraction={CUTOFF_FRACTION:.2f}, "
        "construct perturbed histories H′, compute π(H), π(H′), and report distributions of |π(H)-π(H′)| "
        "and |logBF(H)-logBF(H′)|. Results are stratified by activity level (volume quantiles) and the truncation "
        "event E_R (whether leading null price rows were removed).",
        "",
        "## Outputs",
        "- `reports/experiment_D_deltas.csv` (per-market deltas)",
        "- `reports/experiment_D_summary.csv` (stratified summary)",
        "- `plots/experiment_D_hist_logBF_*.png`, `plots/experiment_D_hist_pi_*.png`",
        "",
        "## Stratified summary (median and 90th percentile)",
        "",
        to_md(summary),
        "",
        "## Notes",
        "- logBF is computed from π via log-odds (prior odds = 1). Replace logBF_from_pi if your inference engine outputs BF directly.",
        "- Replace posterior_model_predict(...) with your Bayesian posterior π_t when ready.",
        "",
    ]
    report_path = REPORTS_DIR / "experiment_D_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("Done.")
    print(f"- Report:   {report_path}")
    print(f"- Deltas:   {deltas_path}")
    print(f"- Summary:  {summary_path}")
    print(f"- Plots:    {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
