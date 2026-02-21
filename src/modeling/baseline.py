from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BaselineConfig:
    """Rule-based baseline scoring model.

    Score = wq * mean(cs_rank(quality)) + wm * mean(cs_rank(momentum))
    """
    date_col: str = "date"
    score_col: str = "baseline_score"
    quality_features: List[str] | None = None
    momentum_features: List[str] | None = None
    quality_weight: float = 0.5
    momentum_weight: float = 0.5


def cs_rank_pct(df: pd.DataFrame, date_col: str, col: str) -> pd.Series:
    """Cross-sectional percentile rank within each date in [0, 1]."""
    return df.groupby(date_col)[col].rank(pct=True, method="average")


def compute_rule_based_score(df: pd.DataFrame, cfg: BaselineConfig) -> pd.DataFrame:
    if cfg.quality_features is None or cfg.momentum_features is None:
        raise ValueError("Provide quality_features and momentum_features in BaselineConfig.")

    out = df.copy()
    required = [cfg.date_col] + cfg.quality_features + cfg.momentum_features
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    q_ranks = [cs_rank_pct(out, cfg.date_col, c) for c in cfg.quality_features]
    m_ranks = [cs_rank_pct(out, cfg.date_col, c) for c in cfg.momentum_features]

    out["_quality_score"] = np.mean(np.vstack([s.values for s in q_ranks]), axis=0)
    out["_momentum_score"] = np.mean(np.vstack([s.values for s in m_ranks]), axis=0)

    out[cfg.score_col] = cfg.quality_weight * out["_quality_score"] + cfg.momentum_weight * out["_momentum_score"]

    out.drop(columns=["_quality_score", "_momentum_score"], inplace=True)
    return out
