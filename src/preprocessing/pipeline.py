"""
Orchestration utilities for preprocessing + feature engineering.

This module enforces the intended ordering of:
1) Daily preprocessing (sanity + clipping + ffill)
2) End-of-month sampling
3) Monthly cross-sectional winsorization + percentile ranks
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .features_engineering import engineer_features
from .preprocessing import PreprocessConfig, preprocess_panel, winsorize_fundamentals_cs
from .target_construction import TargetConfig, construct_target


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
    preprocess_config: PreprocessConfig,
    *,
    preprocessed: bool = False,
    apply_winsorization: bool = True,
    winsor_lower_q: float = 0.01,
    winsor_upper_q: float = 0.99,
    growth_winsor_lower_q: float = 0.02,
    growth_winsor_upper_q: float = 0.98,
    growth_cols: Optional[Sequence[str]] = None,
    add_target: bool = False,
    target_config: Optional[TargetConfig] = None,
) -> pd.DataFrame:
    """
    Run the ordered preprocessing + feature engineering pipeline and return an EOM panel.

    Args:
        df_raw: Raw daily panel with merged inputs.
        preprocess_config: Configuration for daily preprocessing.
        preprocessed: If True, assume df_raw is already preprocessed.
        apply_winsorization: Apply cross-sectional winsorization on EOM.
        winsor_lower_q: Lower quantile for fundamental winsorization.
        winsor_upper_q: Upper quantile for fundamental winsorization.
        growth_winsor_lower_q: Lower quantile for growth winsorization.
        growth_winsor_upper_q: Upper quantile for growth winsorization.
        growth_cols: Growth columns to winsorize more aggressively.

    Returns:
        Feature-engineered end-of-month panel, optionally with targets.
    """
    df_prep = df_raw if preprocessed else preprocess_panel(df_raw, config=preprocess_config)

    df_eom = _sample_eom(df_prep, preprocess_config.date_col, preprocess_config.ticker_col)
    growth_cols = list(growth_cols) if growth_cols is not None else list(preprocess_config.growth_cols)
    if apply_winsorization:
        df_eom = winsorize_fundamentals_cs(
            df_eom,
            date_col=preprocess_config.date_col,
            fundamental_cols=preprocess_config.fundamental_cols,
            growth_cols=growth_cols,
            lower_q=winsor_lower_q,
            upper_q=winsor_upper_q,
            growth_lower_q=growth_winsor_lower_q,
            growth_upper_q=growth_winsor_upper_q,
        )
    df_eom = engineer_features(
        df_eom,
        date_col=preprocess_config.date_col,
        ticker_col=preprocess_config.ticker_col,
    )
    if add_target:
        df_eom = construct_target(df_eom, config=target_config)
    return df_eom
