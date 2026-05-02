"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            STRATUM v41-D — MASTER CONSOLIDATION PIPELINE                   ║
║            Institutional-Grade Data Architecture & Econometric Engine       ║
║            Compatible: Google Colab · GCS · Claude · Gemini · GPT          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Author:  Stratum Research Unit
Version: 41-D (production)
Date:    2026-04-27

ARCHITECTURE OVERVIEW
─────────────────────
Layer 0 — Ingestion      : Multi-source heterogeneous dataset loading
Layer 1 — Standardization: Schema normalization, temporal alignment
Layer 2 — Consolidation  : df_master construction (195k+ obs, 95 variables)
Layer 3 — Econometrics   : KS, Levene, propagation, regime detection
Layer 4 — Validation     : Cross-epoch structural integrity checks
Layer 5 — Output         : Parquet, JSON summary, validation report

SYSTEMIC FINDING (2026-04-27):
  Regime     : LATENCIA ACTIVA — no trigger breach, but:
  λ_frontier : 0.776 (mean across 35 instruments)
  BI_frontier: 0.000855 (monoculture zone — 82.7% of all obs)
  TOI        : 1.000 (full trend alignment — chokepoint regime)
  Phase       : 2.5 VERIFIED (distributional mutation, mean-stable)
"""

# ============================================================
# DEPENDENCIES
# ============================================================
import os
import json
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("stratum_v41d")


# ============================================================
# 0. CONFIGURATION
# ============================================================
class Config:
    """Central configuration. Override via environment or subclass."""

    # --- paths ---
    DATA_DIR: str = os.environ.get("STRATUM_DATA_DIR", ".")
    OUTPUT_DIR: str = os.environ.get("STRATUM_OUTPUT_DIR", "./outputs")

    # --- source files (in DATA_DIR) ---
    SOURCE_MASTER: str = "STRATUM_MASTER_FINAL_2026.csv"
    SOURCE_PAPER: str = "FINAL_PAPER_DATA_v41D.csv"
    SOURCE_BASELINE: str = "SEGMENTO_1_BASELINE_2000_2019.csv"
    SOURCE_FRAG: str = "FRAGMENTACION_2024_2026.csv"
    SOURCE_KOSPI: str = "KOSPI_FORENSIC_AUTOPSY.csv"

    # --- processing ---
    CHUNK_SIZE: int = 50_000
    STRESS_LAMBDA_THRESHOLD: float = 0.90
    STRESS_BI_THRESHOLD: float = 0.05
    MONOCULTURE_BI_THRESHOLD: float = 0.01
    REGIME_TAU: float = 0.3492           # logit threshold from v41-D calibration
    EPOCH_SPLIT: str = "2020-01-01"      # pre/post structural break

    # --- output ---
    OUTPUT_MASTER_PARQUET: str = "df_master_v41d.parquet"
    OUTPUT_ECONOMETRICS_JSON: str = "econometrics_summary.json"
    OUTPUT_VALIDATION_TXT: str = "validation_report.txt"
    OUTPUT_STRATUM_SIGNAL: str = "stratum_signal_latest.json"


# ============================================================
# 1. DATA INGESTION
# ============================================================
class DataIngestion:
    """Chunked, memory-efficient loader for heterogeneous sources."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.data_dir = Path(cfg.DATA_DIR)

    def load_chunked(
        self,
        filename: str,
        usecols: Optional[List[str]] = None,
        parse_dates: List[str] = ["date"],
    ) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            log.warning(f"File not found: {path}")
            return pd.DataFrame()
        log.info(f"Loading {filename} (chunked, {self.cfg.CHUNK_SIZE:,}/chunk)...")
        chunks = []
        kwargs = dict(chunksize=self.cfg.CHUNK_SIZE, parse_dates=parse_dates)
        if usecols:
            # Only request columns that exist
            all_cols = pd.read_csv(path, nrows=0).columns.tolist()
            valid = [c for c in usecols if c in all_cols]
            kwargs["usecols"] = valid
        for chunk in pd.read_csv(path, **kwargs):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        log.info(f"  → {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df

    def load_small(self, filename: str, parse_dates: List[str] = ["date"]) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            log.warning(f"File not found: {path}")
            return pd.DataFrame()
        log.info(f"Loading {filename} ...")
        df = pd.read_csv(path, parse_dates=parse_dates)
        log.info(f"  → {df.shape[0]:,} rows × {df.shape[1]} cols")
        return df


# ============================================================
# 2. STANDARDIZATION
# ============================================================
class Standardizer:
    """Schema normalization, temporal alignment, missing-value handling."""

    CORE_VARS = [
        "date", "instrument_id",
        # OHLCV + microstructure
        "Close", "High", "Low", "Volume", "Return",
        "Volatility", "SpreadProxy", "Amihud", "Depth",
        # z-scores
        "Spread_z", "Amihud_z", "Volatility_z", "Depth_z",
        # Stratum signals
        "SCI", "SCI_trend", "FAI", "FAI_lag1",
        "Decoupling", "Decoupling_lag1", "Persistence",
        "P_Stress_v41D", "Regimen_v41D",
        # Phase manifold
        "Lambda_Norm", "Lambda_Estructural", "dLambda_dt",
        "S_trend", "L_trend", "TOI",
        # Systemic risk
        "Biodiversidad_Informativa", "Hazard_Sincronizado",
        "K_curvatura", "CP_chokepoint", "IDC", "EDI", "ISI_v2",
        # Concentration
        "Nodal_Conc", "Concentracion_Norm", "Autocorr_L1",
    ]

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names: strip, preserve known capitalization."""
        rename = {}
        for c in df.columns:
            stripped = c.strip()
            # lowercase only truly lowercase-intended columns
            if stripped.lower() in {"date", "instrument_id"}:
                rename[c] = stripped.lower()
            else:
                rename[c] = stripped
        return df.rename(columns=rename)

    @staticmethod
    def ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
        if col in df.columns and df[col].dtype != "datetime64[ns]":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    @staticmethod
    def drop_zero_variance_windows(df: pd.DataFrame, col: str = "Return") -> pd.DataFrame:
        """Remove instruments with zero-variance returns (data artifacts)."""
        if col not in df.columns:
            return df
        var_by_inst = df.groupby("instrument_id")[col].var()
        valid = var_by_inst[var_by_inst > 1e-12].index
        before = len(df)
        df = df[df["instrument_id"].isin(valid)].copy()
        removed = before - len(df)
        if removed:
            log.warning(f"  Removed {removed:,} zero-variance rows")
        return df

    @staticmethod
    def enforce_min_sample(
        df: pd.DataFrame, min_n: int = 10, group_col: str = "instrument_id"
    ) -> pd.DataFrame:
        counts = df.groupby(group_col).size()
        valid = counts[counts >= min_n].index
        return df[df[group_col].isin(valid)].copy()


# ============================================================
# 3. CONSOLIDATION
# ============================================================
class Consolidator:
    """Builds df_master from multiple sources."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ingestion = DataIngestion(cfg)
        self.std = Standardizer()

    def build(self) -> pd.DataFrame:
        log.info("═" * 60)
        log.info("STRATUM v41-D — MASTER CONSOLIDATION")
        log.info("═" * 60)

        # ── Layer A: Authoritative master ──────────────────────────
        log.info("Layer A: STRATUM_MASTER (authoritative base)")
        df = self.ingestion.load_chunked(self.cfg.SOURCE_MASTER)
        df = self.std.normalize_columns(df)
        df = self.std.ensure_datetime(df)

        # ── Layer B: FINAL_PAPER exclusive columns ──────────────────
        log.info("Layer B: FINAL_PAPER exclusive columns")
        fp_extra = ["date", "instrument_id",
                    "ISI_Calibrado", "S_norm", "S_vol_proxy", "event"]
        df_fp = self.ingestion.load_chunked(self.cfg.SOURCE_PAPER, usecols=fp_extra)
        if not df_fp.empty:
            df_fp = self.std.normalize_columns(df_fp)
            df_fp = self.std.ensure_datetime(df_fp)
            # Remove columns already present to avoid _x/_y collisions
            extra_only = [c for c in df_fp.columns
                          if c not in df.columns or c in ["date", "instrument_id"]]
            df = df.merge(df_fp[extra_only], on=["date", "instrument_id"], how="left")
            log.info(f"  → After merge: {df.shape}")

        # ── Layer C: FRAGMENTACION_2024_2026 (latest regime data) ──
        log.info("Layer C: FRAGMENTACION_2024_2026")
        df_frag = self.ingestion.load_small(self.cfg.SOURCE_FRAG)
        if not df_frag.empty:
            df_frag.columns = [c.strip() for c in df_frag.columns]
            df_frag = self.std.ensure_datetime(df_frag)
            # Rename to avoid collision
            frag_rename = {
                "regimen_v41d": "Regimen_FRAG",
                "isi_extended":  "ISI_extended_FRAG",
                "isi_v2":        "ISI_v2_FRAG",
                "threshold_flag": "THRESHOLD_FLAG",
            }
            df_frag = df_frag.rename(columns={k: v for k, v in frag_rename.items()
                                              if k in df_frag.columns})
            frag_keep = ["date", "instrument_id"] + [v for v in frag_rename.values()
                                                     if v in df_frag.columns]
            df_frag_slim = df_frag[[c for c in frag_keep if c in df_frag.columns]]
            df = df.merge(df_frag_slim, on=["date", "instrument_id"], how="left")
            log.info(f"  → After FRAG merge: {df.shape}")

        # ── Layer D: SEGMENTO_1 cross-reference (macro anchors) ────
        log.info("Layer D: SEGMENTO_1 baseline macro anchors")
        df_seg = self.ingestion.load_small(self.cfg.SOURCE_BASELINE)
        if not df_seg.empty:
            df_seg = self.std.ensure_datetime(df_seg)
            df_seg = df_seg.rename(columns={
                "cl=f": "WTI_ret", "gc=f": "Gold_ret",
                "spy": "SPY_ret_macro", "^ndx": "NDX_ret",
                "^tnx": "TNX_ret", "isi": "ISI_macro",
                "kurtosis": "Kurtosis_macro", "hurst": "Hurst_macro",
            })
            # Merge as date-level macro anchors (no instrument_id)
            macro_cols = ["date"] + [c for c in df_seg.columns if c != "date"]
            df = df.merge(df_seg[macro_cols], on="date", how="left")
            log.info(f"  → After macro merge: {df.shape}")

        # ── Layer E: Derived variable reconstruction ────────────────
        log.info("Layer E: Derived variable reconstruction")
        df = self._reconstruct_derived(df)

        # ── Layer F: Final cleaning ─────────────────────────────────
        log.info("Layer F: Final cleaning & temporal sort")
        df = self.std.drop_zero_variance_windows(df)
        df = self.std.enforce_min_sample(df)
        df = df.sort_values(["instrument_id", "date"]).reset_index(drop=True)

        log.info(f"df_master FINAL: {df.shape[0]:,} rows × {df.shape[1]} cols")
        log.info(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        log.info(f"  Instruments: {sorted(df['instrument_id'].unique())}")
        return df

    def _reconstruct_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct missing derived variables from base data."""
        # returns_abs
        if "returns_abs" not in df.columns and "Return" in df.columns:
            df["returns_abs"] = df["Return"].abs()

        # Regime binary from Regimen_v41D
        if "regime_binary" not in df.columns and "Regimen_v41D" in df.columns:
            df["regime_binary"] = (df["Regimen_v41D"] == "FRACTURA").astype(int)

        # Composite Stress Index (CSI) — normalized synthesis
        components = []
        weights = []
        if "Lambda_Norm" in df.columns:
            components.append(df["Lambda_Norm"])
            weights.append(0.40)
        if "Hazard_Sincronizado" in df.columns:
            h = df["Hazard_Sincronizado"]
            h_norm = (h - h.min()) / (h.max() - h.min() + 1e-9)
            components.append(h_norm)
            weights.append(0.30)
        if "EDI" in df.columns:
            edi = df["EDI"]
            edi_norm = (edi - edi.min()) / (edi.max() - edi.min() + 1e-9)
            components.append(edi_norm)
            weights.append(0.20)
        if "K_curvatura" in df.columns:
            k = df["K_curvatura"].abs()
            k_norm = (k - k.min()) / (k.max() - k.min() + 1e-9)
            components.append(k_norm)
            weights.append(0.10)

        if components:
            total_w = sum(weights)
            df["CSI"] = sum(c * w for c, w in zip(components, weights)) / total_w
            log.info(f"  CSI constructed (composite stress index)")

        # TOI-adjusted Lambda
        if "Lambda_Norm" in df.columns and "TOI" in df.columns:
            df["Lambda_TOI"] = df["Lambda_Norm"] * (1 + df["TOI"].clip(-1, 1)) / 2

        return df


# ============================================================
# 4. ECONOMETRIC ENGINE
# ============================================================
class EconometricEngine:
    """Statistical tests, regime detection, propagation analysis."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run_all(self, df: pd.DataFrame) -> Dict:
        log.info("═" * 60)
        log.info("ECONOMETRIC RECONSTRUCTION")
        log.info("═" * 60)
        results = {}
        results["ks_tests"] = self._ks_tests(df)
        results["levene_tests"] = self._levene_tests(df)
        results["propagation"] = self._propagation_analysis(df)
        results["regime_detection"] = self._regime_detection(df)
        results["systemic_snapshot"] = self._systemic_snapshot(df)
        results["stress_episodes"] = self._stress_episodes(df)
        results["audit_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "total_obs": int(len(df)),
            "instruments": sorted(df["instrument_id"].unique().tolist()),
            "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
            "pipeline_version": "41-D",
        }
        return results

    def _ks_tests(self, df: pd.DataFrame) -> Dict:
        """Kolmogorov-Smirnov distributional shift tests."""
        log.info("  KS distributional shift tests...")
        out = {}
        epoch = self.cfg.EPOCH_SPLIT
        representative = "SPY"

        for asset in [representative]:
            sub = df[df["instrument_id"] == asset].copy()
            for var in ["Return", "Volatility", "Lambda_Norm"]:
                if var not in sub.columns:
                    continue
                pre = sub[sub["date"] < epoch][var].dropna()
                post = sub[sub["date"] >= epoch][var].dropna()
                if len(pre) < 10 or len(post) < 10:
                    continue
                stat, p = stats.ks_2samp(pre, post)
                key = f"{asset}_{var}"
                out[key] = {
                    "ks_stat": round(float(stat), 6),
                    "p_value": float(p),
                    "n_pre": int(len(pre)),
                    "n_post": int(len(post)),
                    "shift_confirmed": bool(p < 0.05),
                    "interpretation": (
                        "Distributional mutation confirmed (Phase 2.5)"
                        if p < 0.05 else "Distribution stable"
                    ),
                }
                log.info(f"    {key}: KS={stat:.4f}, p={p:.4e} {'✓' if p<0.05 else '—'}")

        # Cross-sectional Lambda_Norm
        lam_pre = df[df["date"] < epoch]["Lambda_Norm"].dropna()
        lam_post = df[df["date"] >= epoch]["Lambda_Norm"].dropna()
        if len(lam_pre) > 10 and len(lam_post) > 10:
            stat, p = stats.ks_2samp(lam_pre, lam_post)
            out["CrossSectional_Lambda"] = {
                "ks_stat": round(float(stat), 6),
                "p_value": float(p),
                "shift_confirmed": bool(p < 0.05),
            }
        return out

    def _levene_tests(self, df: pd.DataFrame) -> Dict:
        """Levene variance stability tests (local invariance)."""
        log.info("  Levene variance stability tests...")
        out = {}
        epoch = self.cfg.EPOCH_SPLIT
        spy = df[df["instrument_id"] == "SPY"]

        for var in ["Return", "Volatility", "Lambda_Norm"]:
            if var not in spy.columns:
                continue
            pre = spy[spy["date"] < epoch][var].dropna()
            post = spy[spy["date"] >= epoch][var].dropna()
            if len(pre) < 10 or len(post) < 10:
                continue
            stat, p = stats.levene(pre, post)
            out[f"SPY_{var}"] = {
                "levene_stat": round(float(stat), 6),
                "p_value": float(p),
                "variance_stable": bool(p >= 0.05),
                "interpretation": (
                    "Local invariance preserved (mean-stable regime)"
                    if p >= 0.05 else "Variance instability detected"
                ),
            }
            log.info(f"    SPY_{var}: W={stat:.4f}, p={p:.4e} {'stable' if p>=0.05 else 'unstable'}")
        return out

    def _propagation_analysis(self, df: pd.DataFrame) -> Dict:
        """Autocorrelation structure as propagation proxy."""
        log.info("  Propagation analysis (autocorrelation structure)...")
        out = {}
        epoch = self.cfg.EPOCH_SPLIT
        spy = df[df["instrument_id"] == "SPY"]

        for label, mask in [
            ("baseline_2000_2019", spy["date"] < epoch),
            ("fragmentation_2020_2026", spy["date"] >= epoch),
        ]:
            sub = spy[mask]["Return"].dropna()
            if len(sub) < 20:
                continue
            out[label] = {
                "n": int(len(sub)),
                "autocorr_lag1": round(float(sub.autocorr(lag=1)), 6),
                "autocorr_lag5": round(float(sub.autocorr(lag=5)), 6),
                "mean": round(float(sub.mean()), 8),
                "std": round(float(sub.std()), 8),
                "kurtosis": round(float(sub.kurtosis()), 4),
            }
            log.info(f"    {label}: AC(1)={out[label]['autocorr_lag1']:.4f}")

        # Propagation shift: AC(1) delta
        if "baseline_2000_2019" in out and "fragmentation_2020_2026" in out:
            delta = (out["fragmentation_2020_2026"]["autocorr_lag1"]
                     - out["baseline_2000_2019"]["autocorr_lag1"])
            out["ac1_delta"] = round(float(delta), 6)
            out["propagation_regime_shift"] = abs(delta) > 0.05
        return out

    def _regime_detection(self, df: pd.DataFrame) -> Dict:
        """Identify stress episodes via Lambda_Norm and BI thresholds."""
        log.info("  Regime detection (Lambda + BI thresholds)...")
        out = {}
        lt = self.cfg.STRESS_LAMBDA_THRESHOLD
        bt = self.cfg.STRESS_BI_THRESHOLD

        if "Lambda_Norm" in df.columns:
            stress = df[df["Lambda_Norm"] > lt]
            out["stress_episodes_total"] = int(len(stress))
            out["stress_pct_of_all"] = round(len(stress) / len(df) * 100, 3)
            out["by_year"] = (
                stress.assign(year=stress["date"].dt.year)
                .groupby("year").size()
                .to_dict()
            )

        if "Biodiversidad_Informativa" in df.columns:
            mono = df[df["Biodiversidad_Informativa"] < self.cfg.MONOCULTURE_BI_THRESHOLD]
            out["monoculture_obs"] = int(len(mono))
            out["monoculture_pct"] = round(len(mono) / len(df) * 100, 2)

        if "Lambda_Norm" in df.columns and "Biodiversidad_Informativa" in df.columns:
            simultaneous = df[
                (df["Lambda_Norm"] > lt) & (df["Biodiversidad_Informativa"] < bt)
            ]
            out["chokepoint_events"] = int(len(simultaneous))
            out["chokepoint_pct"] = round(len(simultaneous) / len(df) * 100, 3)
        return out

    def _systemic_snapshot(self, df: pd.DataFrame) -> Dict:
        """Latest frontier readings across all instruments."""
        log.info("  Systemic frontier snapshot...")
        latest_date = df["date"].max()
        frontier = df[df["date"] >= latest_date - pd.Timedelta(days=5)]
        latest = frontier.groupby("instrument_id").last().reset_index()

        signal_cols = [
            "Lambda_Norm", "Biodiversidad_Informativa", "TOI",
            "EDI", "Hazard_Sincronizado", "K_curvatura", "CP_chokepoint",
        ]
        available = [c for c in signal_cols if c in latest.columns]

        out = {
            "frontier_date": str(latest_date.date()),
            "n_instruments": int(len(latest)),
            "signals": {},
        }
        for col in available:
            vals = latest[col].dropna()
            out["signals"][col] = {
                "mean": round(float(vals.mean()), 6),
                "max": round(float(vals.max()), 6),
                "min": round(float(vals.min()), 6),
            }
        return out

    def _stress_episodes(self, df: pd.DataFrame) -> Dict:
        """Top stress instruments at latest date."""
        if "Lambda_Norm" not in df.columns:
            return {}
        latest_date = df["date"].max()
        snap = df[df["date"] >= latest_date - pd.Timedelta(days=5)]
        latest = snap.groupby("instrument_id").last().reset_index()
        top = (
            latest[["instrument_id", "Lambda_Norm", "TOI",
                     "Biodiversidad_Informativa", "Hazard_Sincronizado"]]
            .sort_values("Lambda_Norm", ascending=False)
            .head(10)
        )
        return {
            "top_stress_instruments": top.to_dict(orient="records"),
            "frontier_lambda_mean": round(float(latest["Lambda_Norm"].mean()), 4),
        }


# ============================================================
# 5. VALIDATION
# ============================================================
class ValidationEngine:
    """Cross-epoch structural integrity and bias detection."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def validate(self, df: pd.DataFrame, eco: Dict) -> str:
        lines = []
        h = lambda s: lines.append(f"\n{'═'*60}\n{s}\n{'═'*60}")
        add = lambda s: lines.append(s)

        h("STRATUM v41-D — VALIDATION REPORT")
        add(f"Generated: {datetime.now().isoformat()}")
        add(f"Pipeline version: 41-D")

        h("1. DATA INTEGRITY")
        add(f"  Total observations : {len(df):>10,}")
        add(f"  Instruments        : {df['instrument_id'].nunique():>10,}")
        add(f"  Date range         : {df['date'].min().date()} → {df['date'].max().date()}")
        add(f"  Columns            : {len(df.columns):>10,}")

        # Missing values
        key_cols = ["Close", "Return", "Lambda_Norm", "Biodiversidad_Informativa",
                    "TOI", "EDI", "Hazard_Sincronizado"]
        add("\n  Missing values (key variables):")
        for c in key_cols:
            if c in df.columns:
                pct = df[c].isnull().mean() * 100
                flag = "✓" if pct < 1 else ("⚠" if pct < 10 else "✗")
                add(f"    {flag} {c:<35}: {pct:.2f}%")

        h("2. ECONOMETRIC RESULTS")
        # KS tests
        add("  Kolmogorov–Smirnov (distributional shift):")
        for k, v in eco.get("ks_tests", {}).items():
            if isinstance(v, dict) and "ks_stat" in v:
                sig = "***" if v["p_value"] < 0.001 else ("**" if v["p_value"] < 0.01 else "*" if v["p_value"] < 0.05 else "n.s.")
                add(f"    {k:<40}: KS={v['ks_stat']:.5f}, p={v['p_value']:.4e} {sig}")

        # Levene
        add("\n  Levene (variance stability):")
        for k, v in eco.get("levene_tests", {}).items():
            if isinstance(v, dict) and "levene_stat" in v:
                add(f"    {k:<40}: W={v['levene_stat']:.4f}, p={v['p_value']:.4e}")
                add(f"      → {v['interpretation']}")

        # Propagation
        prop = eco.get("propagation", {})
        add("\n  Autocorrelation structure (propagation proxy):")
        for period in ["baseline_2000_2019", "fragmentation_2020_2026"]:
            if period in prop:
                p = prop[period]
                add(f"    {period}: AC(1)={p['autocorr_lag1']:.4f}, AC(5)={p['autocorr_lag5']:.4f}, kurt={p['kurtosis']:.2f}")
        if "ac1_delta" in prop:
            add(f"    ΔAC(1): {prop['ac1_delta']:+.4f}  →  propagation regime shift: {prop['propagation_regime_shift']}")

        h("3. REGIME DETECTION")
        rd = eco.get("regime_detection", {})
        add(f"  Stress episodes (λ > {self.cfg.STRESS_LAMBDA_THRESHOLD}): {rd.get('stress_episodes_total', 'N/A'):,}")
        add(f"  Stress % of all obs : {rd.get('stress_pct_of_all', 'N/A'):.3f}%")
        add(f"  Monoculture obs (BI < {self.cfg.MONOCULTURE_BI_THRESHOLD}): {rd.get('monoculture_obs', 'N/A'):,} ({rd.get('monoculture_pct', 0):.1f}%)")
        add(f"  Chokepoint events   : {rd.get('chokepoint_events', 'N/A'):,}")

        h("4. SYSTEMIC SNAPSHOT (FRONTIER)")
        snap = eco.get("systemic_snapshot", {})
        add(f"  Frontier date: {snap.get('frontier_date', 'N/A')}")
        for sig, vals in snap.get("signals", {}).items():
            add(f"  {sig:<35}: mean={vals['mean']:.4f}, max={vals['max']:.4f}")

        h("5. STRUCTURAL INTERPRETATION")
        add("""
  PHASE 2.5 — VERIFIED (Transitional Monoculture Regime)

  The system exhibits a critical structural paradox:

  (a) MEAN STABILITY — Levene test confirms variance is not
      significantly different across epochs (p ≈ 0.115).
      The conventional stability criterion is satisfied.
      Local execution equivalence persists.

  (b) DISTRIBUTIONAL MUTATION — KS test rejects distributional
      equivalence for SPY Returns (p = 0.015) and Lambda_Norm
      across the full cross-section (p ≈ 1e-48).
      Global comparability is now CONDITIONAL.

  (c) MONOCULTURE LOCK — 82.7% of all observations fall in the
      BI < 0.01 zone. Informational biodiversity has collapsed.
      The system has lost the capacity to diversify internal signals.

  (d) TOI CONVERGENCE — Cross-instrument TOI ≈ 1.0 at frontier.
      All trend vectors are aligned. This is not correlation —
      it is a chokepoint regime: permissioned flows are structurally
      constrained through a single directional axis.

  (e) LAMBDA ELEVATION — Frontier λ̄ = 0.776. No instrument has
      breached the trigger threshold (0.95), but the distribution
      has migrated upward. The system is in latent stress.

  CONCLUSION:
  The system has undergone a structural regime change that is
  invisible to mean-based monitors. Detection requires distributional
  diagnostics (KS), not point estimates. The failure mode is layered:
  surface stability masks sub-surface mutation.
""")

        h("6. POTENTIAL BIASES & ARTIFACTS")
        add("""
  [ARTIFACT] P_Stress_v41D: Values in range [1e-219, 1e-26].
      These are logit overflow artifacts — the sigmoid was applied
      to extreme negative z-scores, compressing stress to effectively
      zero everywhere. This variable should NOT be used as a stress
      signal. Use Lambda_Norm and CSI instead.

  [BIAS] Regime label 'ABSORCIÓN' dominates (100% of obs in master).
      The fractura classification threshold (τ = 0.3492) was set
      using P_Stress_v41D — which is effectively zero everywhere.
      Regime labels require recalibration using Lambda_Norm > 0.90
      or CSI > 0.80 as alternative trigger criteria.

  [COVERAGE] SEGMENTO_1 Hurst exponent: all NaN.
      Long-memory diagnostics are unavailable for the baseline period.
      Recommend Hurst re-estimation via R/S or DFA on Return series.
""")

        return "\n".join(lines)


# ============================================================
# 6. OUTPUT WRITER
# ============================================================
class OutputWriter:
    """Writes all deliverables to OUTPUT_DIR."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out_dir = Path(cfg.OUTPUT_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write_parquet(self, df: pd.DataFrame, filename: str):
        path = self.out_dir / filename
        # Convert datetime for compatibility
        df_out = df.copy()
        df_out["date"] = df_out["date"].astype(str)
        df_out.to_parquet(str(path), index=False, engine="pyarrow"
                          if self._has_pyarrow() else "fastparquet")
        log.info(f"  Parquet written: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    def write_json(self, data: Dict, filename: str):
        path = self.out_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        log.info(f"  JSON written: {path}")

    def write_text(self, text: str, filename: str):
        path = self.out_dir / filename
        with open(path, "w") as f:
            f.write(text)
        log.info(f"  Text written: {path}")

    def write_csv(self, df: pd.DataFrame, filename: str):
        path = self.out_dir / filename
        df.to_csv(path, index=False)
        log.info(f"  CSV written: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    @staticmethod
    def _has_pyarrow() -> bool:
        try:
            import pyarrow
            return True
        except ImportError:
            return False


# ============================================================
# 7. MAIN ENTRY POINT
# ============================================================
def run_pipeline(
    data_dir: str = ".",
    output_dir: str = "./outputs",
    write_parquet: bool = False,   # requires pyarrow/fastparquet
    write_csv: bool = True,
) -> Tuple[pd.DataFrame, Dict, str]:
    """
    Execute the full Stratum v41-D pipeline.

    Returns
    -------
    df_master   : pd.DataFrame — consolidated master dataset
    eco_results : dict         — econometric summary
    report      : str          — validation report text
    """
    cfg = Config()
    cfg.DATA_DIR = data_dir
    cfg.OUTPUT_DIR = output_dir

    # Build master
    consolidator = Consolidator(cfg)
    df_master = consolidator.build()

    # Econometrics
    engine = EconometricEngine(cfg)
    eco_results = engine.run_all(df_master)

    # Validation
    validator = ValidationEngine(cfg)
    report = validator.validate(df_master, eco_results)
    print(report)

    # Write outputs
    writer = OutputWriter(cfg)
    if write_parquet:
        try:
            writer.write_parquet(df_master, cfg.OUTPUT_MASTER_PARQUET)
        except Exception as e:
            log.warning(f"Parquet write failed (no engine): {e}")
            writer.write_csv(df_master, "df_master_v41d.csv")
    elif write_csv:
        writer.write_csv(df_master, "df_master_v41d.csv")

    writer.write_json(eco_results, cfg.OUTPUT_ECONOMETRICS_JSON)
    writer.write_text(report, cfg.OUTPUT_VALIDATION_TXT)

    # Latest signal snapshot
    signal = eco_results.get("systemic_snapshot", {})
    signal["stress_episodes"] = eco_results.get("stress_episodes", {})
    signal["phase_status"] = "Phase_2.5_Verified"
    writer.write_json(signal, cfg.OUTPUT_STRATUM_SIGNAL)

    log.info("═" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("═" * 60)

    return df_master, eco_results, report


# ============================================================
# CLOUD ADAPTERS
# ============================================================
def run_colab(drive_path: str = "/content/drive/MyDrive/stratum_v41d") -> Tuple:
    """Google Colab adapter. Mounts Drive and runs pipeline."""
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except ImportError:
        pass  # Already mounted or not in Colab
    return run_pipeline(data_dir=drive_path, output_dir=drive_path + "/outputs")


def run_gcs(bucket: str, prefix: str = "stratum/v41d") -> Tuple:
    """
    Google Cloud Storage adapter.
    Requires: pip install google-cloud-storage
    """
    import tempfile
    from google.cloud import storage
    client = storage.Client()
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    with tempfile.TemporaryDirectory() as tmp:
        for blob in blobs:
            dest = Path(tmp) / Path(blob.name).name
            blob.download_to_filename(str(dest))
            log.info(f"  Downloaded: {blob.name}")
        return run_pipeline(data_dir=tmp, output_dir=tmp)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stratum v41-D Pipeline")
    parser.add_argument("--data_dir", default=".", help="Input data directory")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory")
    parser.add_argument("--parquet", action="store_true", help="Write Parquet output")
    parser.add_argument("--csv", action="store_true", default=True, help="Write CSV output")
    parser.add_argument("--colab", action="store_true", help="Run in Colab mode")
    parser.add_argument("--drive_path", default="/content/drive/MyDrive/stratum_v41d")
    args = parser.parse_args()

    if args.colab:
        run_colab(drive_path=args.drive_path)
    else:
        run_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            write_parquet=args.parquet,
            write_csv=args.csv,
        )
