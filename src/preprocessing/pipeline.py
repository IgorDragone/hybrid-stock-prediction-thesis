"""
Orchestration utilities for preprocessing + feature engineering.

This module enforces the intended ordering of:
1) Daily preprocessing (sanity + clipping + ffill)
2) End-of-month sampling
3) Monthly cross-sectional winsorization + percentile ranks
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .features_engineering import engineer_features
from .preprocessing import PreprocessConfig, preprocess_daily_panel, winsorize_fundamentals_cs
from .target_construction import TargetConfig, construct_target


def select_model_ready_columns(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    include_returns: bool = True,
) -> pd.DataFrame:
    """
    Select a model-ready subset of columns and drop raw inputs.
    """
    base_cols = [date_col, ticker_col]
    macro_cols = [
        "stress_index",
        "macro_regime",
        "slope_10y2y",
    ]
    fund_pr_cols = [
        "roe_pr",
        "roa_pr",
        "operating_margin_pr",
        "gross_margin_pr",
        "revenue_growth_pr",
        "earnings_growth_pr",
        "delta_roe_pr",
        "debt_pr",
        "interest_coverage_pr",
        "asset_turnover_pr",
        "fcf_assets_pr",
    ]
    tech_pr_cols = [
        "mom12_pr",
        "mom6_pr",
        "mom3_pr",
        "trend_ratio_pr",
        "vol_pr",
        "vol_ratio_pr",
    ]
    target_cols = ["target_3m", "target_1m"]
    return_cols = []
    if include_returns:
        return_cols = ["fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m"]

    keep = [
        c for c in base_cols + macro_cols + fund_pr_cols + tech_pr_cols + target_cols + return_cols
        if c in df.columns
    ]
    return df[keep].copy()


def _sample_eom(
    df: pd.DataFrame, date_col: str, ticker_col: str
) -> pd.DataFrame:
    df = df.copy()
    df["_month"] = pd.to_datetime(df[date_col]).dt.to_period("M")
    out = (
        df.sort_values([ticker_col, date_col])
          .groupby([ticker_col, "_month"], group_keys=False)
          .tail(1)
          .copy()
    )
    return out.drop(columns=["_month"])


def build_feature_panel(
    df_raw: pd.DataFrame,
    preprocess_config: Optional[PreprocessConfig] = None,
    *,
    preprocessed: bool = False,
    add_target: bool = False,
    target_config: Optional[TargetConfig] = None,
    drop_raws: bool = False,
    save_stages: bool = False,
    stages_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Run the ordered preprocessing + feature engineering pipeline and return an EOM panel.

    Args:
        df_raw: Raw daily panel with merged inputs.
        preprocess_config: Configuration for daily preprocessing.
        preprocessed: If True, assume df_raw is already preprocessed.
        drop_raws: If True, return only model-ready columns.
        save_stages: If True, save intermediate panels to stages_dir.
        stages_dir: Directory where stage parquet files are written.
    Returns:
        Feature-engineered end-of-month panel, optionally with targets.
    """
    stages_path: Optional[Path] = None
    if save_stages:
        if stages_dir is None:
            raise ValueError("stages_dir must be provided when save_stages=True.")
        stages_path = Path(stages_dir)
        stages_path.mkdir(parents=True, exist_ok=True)

    cfg = preprocess_config or PreprocessConfig()
    df_prep = df_raw if preprocessed else preprocess_daily_panel(df_raw, config=cfg)

    if stages_path is not None:
        if not preprocessed:
            df_raw.to_parquet(stages_path / "panel_raw.parquet", index=False)
        df_prep.to_parquet(stages_path / "panel_preprocessed.parquet", index=False)

    df_eom = _sample_eom(df_prep, cfg.date_col, cfg.ticker_col)
    if stages_path is not None:
        df_eom.to_parquet(stages_path / "panel_eom.parquet", index=False)

    growth_cols = list(cfg.growth_cols)
    if cfg.winsorize_fundamentals_cs:
        df_eom = winsorize_fundamentals_cs(
            df_eom,
            date_col=cfg.date_col,
            fundamental_cols=cfg.fundamental_cols,
            growth_cols=growth_cols,
            lower_q=cfg.winsor_lower_q,
            upper_q=cfg.winsor_upper_q,
            growth_lower_q=cfg.growth_winsor_lower_q,
            growth_upper_q=cfg.growth_winsor_upper_q,
        )

    df_eom = engineer_features(
        df_eom,
        date_col=cfg.date_col,
        ticker_col=cfg.ticker_col,
    )
    if stages_path is not None:
        df_eom.to_parquet(stages_path / "panel_features.parquet", index=False)

    if add_target:
        df_eom = construct_target(df_eom, config=target_config)
        if stages_path is not None:
            df_model_ready = select_model_ready_columns(df_eom)
            df_model_ready.to_parquet(stages_path / "panel_model_ready.parquet", index=False)
    
    if drop_raws:
        df_eom = select_model_ready_columns(df_eom)

    return df_eom
