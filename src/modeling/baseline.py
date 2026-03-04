from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BaselineConfig:
    """Baseline scoring configuration."""
    date_col: str = "date"
    score_col: str = "baseline_score"
    momentum_col: str = "mom12_pr"


def add_momentum_score(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    """Baseline: use mom12_pr as the score."""
    if cfg.momentum_col not in df.columns:
        raise ValueError(f"Missing column: {cfg.momentum_col}")
    out = df.copy()
    out[cfg.score_col] = out[cfg.momentum_col]
    return out
