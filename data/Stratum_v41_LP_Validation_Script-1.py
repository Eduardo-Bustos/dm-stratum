#!/usr/bin/env python3
"""
Stratum v41 Local Projection Validation Script
------------------------------------------------
Purpose:
    Estimate impulse responses of the Synchronization Gap (SG) to systemic shocks,
    weighted by a Systemic Relevance Filter (SRF), using local projections.

Inputs:
    A CSV file exported from regime_v41 / Stratum Engine. The script attempts to
    auto-detect common columns such as SG, ISI, FAI, Hurst, VIX, date, shock, and SRF.

Example:
    python Stratum_v41_LP_Validation_Script.py \
        --input regime_v41.csv \
        --date-col date \
        --sg-col SG \
        --shock-col hormuz_shock \
        --srf 1.0 \
        --controls ISI FAI Hurst VIX \
        --horizon 20 \
        --lags 5 \
        --output-dir lp_output

Notes:
    - If no shock column is supplied, the script creates a shock proxy from large
      positive moves in CP / SG using a z-score threshold.
    - Confidence intervals use Newey-West / HAC standard errors.
    - Output includes a CSV of IRF coefficients and publication-ready PNG/PDF figures.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


COMMON_DATE = ["date", "Date", "timestamp", "Timestamp", "day", "time"]
COMMON_SG = ["SG", "sg", "Synchronization Gap", "synchronization_gap", "sync_gap"]
COMMON_ISI = ["ISI", "isi", "Information Saturation Index", "information_saturation_index"]
COMMON_FAI = ["FAI", "fai", "Friction Accumulation Index", "friction_accumulation_index"]
COMMON_HURST = ["Hurst", "hurst", "H", "hurst_exponent"]
COMMON_VIX = ["VIX", "vix", "volatility", "Volatility"]
COMMON_CP = ["CP", "cp", "chokepoint_pressure", "Chokepoint Pressure"]
COMMON_SRF = ["SRF", "srf", "systemic_relevance", "Systemic Relevance"]
COMMON_SHOCK = ["shock", "Shock", "hormuz_shock", "gpr_shock", "cp_shock"]


def first_existing(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    colset = list(columns)
    lowered = {c.lower(): c for c in colset}
    for cand in candidates:
        if cand in colset:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def add_lags(df: pd.DataFrame, cols: list[str], lags: int) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        for lag in range(1, lags + 1):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def infer_controls(df: pd.DataFrame, explicit: Optional[list[str]]) -> list[str]:
    if explicit:
        return [c for c in explicit if c in df.columns]
    inferred = []
    for candidates in [COMMON_ISI, COMMON_FAI, COMMON_HURST, COMMON_VIX]:
        c = first_existing(df.columns, candidates)
        if c and c not in inferred:
            inferred.append(c)
    return inferred


def build_shock(df: pd.DataFrame, shock_col: Optional[str], cp_col: Optional[str], sg_col: str, threshold: float) -> pd.Series:
    if shock_col and shock_col in df.columns:
        return pd.to_numeric(df[shock_col], errors="coerce")
    detected = first_existing(df.columns, COMMON_SHOCK)
    if detected:
        return pd.to_numeric(df[detected], errors="coerce")
    base_col = cp_col if cp_col and cp_col in df.columns else sg_col
    dz = zscore(pd.to_numeric(df[base_col], errors="coerce").diff())
    return (dz > threshold).astype(float)


def local_projection(
    df: pd.DataFrame,
    sg_col: str,
    z_col: str,
    controls: list[str],
    horizon: int,
    lags: int,
    hac_lags: Optional[int] = None,
) -> pd.DataFrame:
    lag_base = [sg_col, z_col] + controls
    work = add_lags(df, lag_base, lags)
    lag_cols = [f"{c}_lag{j}" for c in lag_base for j in range(1, lags + 1) if f"{c}_lag{j}" in work.columns]

    results = []
    for h in range(horizon + 1):
        y_col = f"{sg_col}_lead{h}"
        work[y_col] = work[sg_col].shift(-h)
        rhs = [z_col] + lag_cols
        model_df = work[[y_col] + rhs].replace([np.inf, -np.inf], np.nan).dropna()
        if len(model_df) < max(25, len(rhs) + 5):
            results.append({"horizon": h, "beta": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "nobs": len(model_df)})
            continue
        X = sm.add_constant(model_df[rhs], has_constant="add")
        y = model_df[y_col]
        maxlags = hac_lags if hac_lags is not None else max(1, h + 1)
        fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        results.append({
            "horizon": h,
            "beta": fit.params.get(z_col, np.nan),
            "se": fit.bse.get(z_col, np.nan),
            "t": fit.tvalues.get(z_col, np.nan),
            "p": fit.pvalues.get(z_col, np.nan),
            "nobs": int(fit.nobs),
        })
    out = pd.DataFrame(results)
    out["ci_low_90"] = out["beta"] - 1.645 * out["se"]
    out["ci_high_90"] = out["beta"] + 1.645 * out["se"]
    out["ci_low_95"] = out["beta"] - 1.960 * out["se"]
    out["ci_high_95"] = out["beta"] + 1.960 * out["se"]
    return out


def plot_irf(irf: pd.DataFrame, output_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0, linewidth=1)
    ax.plot(irf["horizon"], irf["beta"], linewidth=2, marker="o", label="IRF")
    ax.fill_between(irf["horizon"], irf["ci_low_90"], irf["ci_high_90"], alpha=0.25, label="90% CI")
    ax.fill_between(irf["horizon"], irf["ci_low_95"], irf["ci_high_95"], alpha=0.15, label="95% CI")
    ax.set_title(title)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Response of SG")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "stratum_lp_irf.png", dpi=240)
    fig.savefig(output_dir / "stratum_lp_irf.pdf")
    plt.close(fig)


def plot_regime_map(df: pd.DataFrame, sg_col: str, fai_col: Optional[str], tau: float, output_dir: Path) -> None:
    if not fai_col or fai_col not in df.columns:
        return
    work = df[[sg_col, fai_col]].dropna()
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(work[sg_col], work[fai_col], s=20, alpha=0.65)
    ax.axvline(tau, linestyle="--", linewidth=1.2)
    ax.set_title("Regime Map: Synchronization Gap vs Friction Accumulation")
    ax.set_xlabel("Synchronization Gap (SG)")
    ax.set_ylabel("Friction Accumulation Index (FAI)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "stratum_regime_map.png", dpi=240)
    fig.savefig(output_dir / "stratum_regime_map.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Stratum v41 local projection IRFs.")
    parser.add_argument("--input", required=True, help="Path to regime_v41 CSV file.")
    parser.add_argument("--output-dir", default="lp_output", help="Directory for outputs.")
    parser.add_argument("--date-col", default=None, help="Date column. Auto-detected if omitted.")
    parser.add_argument("--sg-col", default=None, help="Synchronization Gap column. Auto-detected if omitted.")
    parser.add_argument("--shock-col", default=None, help="Shock column. If omitted, a shock proxy is generated.")
    parser.add_argument("--cp-col", default=None, help="Chokepoint pressure column used for proxy shocks.")
    parser.add_argument("--srf-col", default=None, help="Column containing SRF weights.")
    parser.add_argument("--srf", type=float, default=1.0, help="Constant SRF multiplier if no SRF column is used.")
    parser.add_argument("--controls", nargs="*", default=None, help="Control columns. Auto-detected if omitted.")
    parser.add_argument("--horizon", type=int, default=20, help="Maximum IRF horizon.")
    parser.add_argument("--lags", type=int, default=5, help="Number of lags for SG, shock, and controls.")
    parser.add_argument("--hac-lags", type=int, default=None, help="HAC max lags. Defaults to h+1 per horizon.")
    parser.add_argument("--shock-threshold", type=float, default=1.5, help="Z-score threshold for proxy shock creation.")
    parser.add_argument("--tau", type=float, default=0.3492, help="Stratum v41 structural threshold.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")

    date_col = args.date_col or first_existing(df.columns, COMMON_DATE)
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    sg_col = args.sg_col or first_existing(df.columns, COMMON_SG)
    if not sg_col or sg_col not in df.columns:
        raise ValueError("Could not identify SG column. Provide --sg-col.")
    df[sg_col] = pd.to_numeric(df[sg_col], errors="coerce")

    cp_col = args.cp_col or first_existing(df.columns, COMMON_CP)
    shock_raw = build_shock(df, args.shock_col, cp_col, sg_col, args.shock_threshold)

    srf_col = args.srf_col or first_existing(df.columns, COMMON_SRF)
    if srf_col and srf_col in df.columns:
        srf = pd.to_numeric(df[srf_col], errors="coerce").fillna(args.srf)
    else:
        srf = pd.Series(args.srf, index=df.index)

    df["stratum_weighted_shock"] = shock_raw.fillna(0) * srf
    controls = infer_controls(df, args.controls)
    for c in controls:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    irf = local_projection(
        df=df,
        sg_col=sg_col,
        z_col="stratum_weighted_shock",
        controls=controls,
        horizon=args.horizon,
        lags=args.lags,
        hac_lags=args.hac_lags,
    )
    irf.to_csv(output_dir / "stratum_lp_irf_coefficients.csv", index=False)

    title = "Stratum v41 Local Projection IRF: SG response to SRF-weighted shock"
    plot_irf(irf, output_dir, title)

    fai_col = first_existing(df.columns, COMMON_FAI)
    plot_regime_map(df, sg_col, fai_col, args.tau, output_dir)

    summary = {
        "input": str(input_path),
        "sg_col": sg_col,
        "date_col": date_col,
        "cp_col": cp_col,
        "srf_col": srf_col,
        "controls": controls,
        "horizon": args.horizon,
        "lags": args.lags,
        "tau": args.tau,
        "n_rows": int(len(df)),
        "n_weighted_shocks_nonzero": int((df["stratum_weighted_shock"] != 0).sum()),
    }
    pd.Series(summary).to_csv(output_dir / "stratum_lp_run_summary.csv")
    print("Done. Outputs written to:", output_dir.resolve())
    print(pd.Series(summary).to_string())


if __name__ == "__main__":
    main()
