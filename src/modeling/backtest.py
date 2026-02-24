from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    date_col: str = "date"
    ticker_col: str = "ticker"
    ret_1m_col: str = "fwd_ret_1m"
    top_k: int = 5
    overlay_enabled: bool = False
    stress_col: str = "stress_index"
    stress_threshold: float = 0.5
    risk_off_exposure: float = 0.6
    cash_return: float = 0.0


def _top_k_by_score(x: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    x = x.dropna(subset=[score_col])
    return x.sort_values(score_col, ascending=False).head(k)


def _portfolio_returns(
    df_scores: pd.DataFrame,
    score_col: str,
    cfg: BacktestConfig,
) -> pd.Series:
    d = df_scores.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])

    for c in [cfg.date_col, cfg.ticker_col, cfg.ret_1m_col, score_col]:
        if c not in d.columns:
            raise ValueError(f"Missing required column: {c}")

    port = (
        d.groupby(cfg.date_col, group_keys=False)
        .apply(lambda x: _top_k_by_score(x, score_col, cfg.top_k)[cfg.ret_1m_col].mean())
    )
    port.name = "port_ret"
    return _apply_overlay(port, d, cfg)


def _apply_overlay(
    port_ret: pd.Series, df_scores: pd.DataFrame, cfg: BacktestConfig
) -> pd.Series:
    if not cfg.overlay_enabled:
        return port_ret
    if cfg.stress_col not in df_scores.columns:
        raise ValueError(f"Missing required column for overlay: {cfg.stress_col}")

    stress = df_scores.groupby(cfg.date_col)[cfg.stress_col].first()
    exposure = np.where(stress > cfg.stress_threshold, cfg.risk_off_exposure, 1.0)
    exposure = pd.Series(exposure, index=stress.index).reindex(port_ret.index)
    return exposure * port_ret + (1.0 - exposure) * cfg.cash_return


def equal_weight_returns(df: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    """Equal-weight universe return (monthly mean of fwd_ret_1m)."""
    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])
    if cfg.ret_1m_col not in d.columns:
        raise ValueError(f"Missing required column: {cfg.ret_1m_col}")
    port = d.groupby(cfg.date_col)[cfg.ret_1m_col].mean()
    port.name = "port_ret"
    return _apply_overlay(port, d, cfg)


def _turnover(top_k_df: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    top_k_df = top_k_df.sort_values([cfg.date_col, cfg.ticker_col])
    top_k_df["month"] = top_k_df[cfg.date_col].dt.to_period("M")
    by_month = top_k_df.groupby("month")[cfg.ticker_col].apply(set)

    turnovers = []
    prev = None
    for m, names in by_month.items():
        if prev is None:
            turnovers.append(0.0)
        else:
            overlap = len(prev & names)
            turnovers.append(1.0 - overlap / cfg.top_k)
        prev = names
    return pd.Series(turnovers, index=by_month.index, name="turnover")


def _equity_curve(port_ret: pd.Series) -> pd.Series:
    return (1.0 + port_ret.fillna(0.0)).cumprod()


def _hit_rate(
    df_scores: pd.DataFrame,
    score_col: str,
    cfg: BacktestConfig,
) -> float:
    d = df_scores.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])

    def _month_hit(x: pd.DataFrame) -> float:
        pred = _top_k_by_score(x, score_col, cfg.top_k)[cfg.ticker_col].tolist()
        actual = _top_k_by_score(x, cfg.ret_1m_col, cfg.top_k)[cfg.ticker_col].tolist()
        if not pred:
            return np.nan
        return len(set(pred) & set(actual)) / cfg.top_k

    hit = d.groupby(cfg.date_col).apply(_month_hit)
    return float(hit.mean())


def backtest_from_scores(
    df_scores: pd.DataFrame,
    score_cols: Dict[str, str],
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Monthly long-only top-K backtest for multiple score columns."""
    summary_rows = []
    artifacts: Dict[str, pd.DataFrame] = {}

    for model_name, score_col in score_cols.items():
        d = df_scores.copy()
        d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])

        top_k_rows = (
            d.groupby(cfg.date_col, group_keys=False)
            .apply(lambda x: _top_k_by_score(x, score_col, cfg.top_k))
        )
        port_ret = _portfolio_returns(d, score_col, cfg)
        eq = _equity_curve(port_ret)
        to = _turnover(top_k_rows, cfg)

        art = pd.DataFrame({"port_ret": port_ret, "equity": eq}).dropna(subset=["port_ret"])
        art = art.join(to, how="left")
        artifacts[model_name] = art

        r = art["port_ret"]
        e = art["equity"]
        max_dd = (e / e.cummax() - 1.0).min()

        vol = r.std()
        sharpe = (r.mean() / vol) * np.sqrt(12) if vol != 0 else np.nan
        cagr = (e.iloc[-1] ** (12 / len(e)) - 1.0) if len(e) > 1 else np.nan

        summary_rows.append({
            "model": model_name,
            "mean_monthly_return": float(r.mean()),
            "vol_monthly": float(vol),
            "sharpe": float(sharpe),
            "cagr": float(cagr),
            "hit_rate": _hit_rate(d, score_col, cfg),
            "max_drawdown": float(max_dd),
            "mean_turnover": float(art["turnover"].mean()),
        })

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return summary, artifacts
