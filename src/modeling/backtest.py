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


def _top_k_panel(
    df_scores: pd.DataFrame,
    score_col: str,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    return (
        df_scores.groupby(cfg.date_col, group_keys=False)
        .apply(lambda x: _top_k_by_score(x, score_col, cfg.top_k))
    )


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


def _portfolio_returns(
    df_scores: pd.DataFrame,
    score_col: str,
    cfg: BacktestConfig,
    top_k_rows: pd.DataFrame,
) -> pd.Series:
    scores = df_scores.copy()
    scores[cfg.date_col] = pd.to_datetime(scores[cfg.date_col])

    for c in [cfg.date_col, cfg.ticker_col, cfg.ret_1m_col, score_col]:
        if c not in scores.columns:
            raise ValueError(f"Missing required column: {c}")

    port = top_k_rows.groupby(cfg.date_col)[cfg.ret_1m_col].mean()
    port.name = "port_ret"
    return _apply_overlay(port, scores, cfg)


def _turnover(top_k_df: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    top_k_df = top_k_df.sort_values([cfg.date_col, cfg.ticker_col])
    by_date = top_k_df.groupby(cfg.date_col)[cfg.ticker_col].apply(set)

    turnovers = []
    prev = None
    for _, names in by_date.items():
        if prev is None:
            turnovers.append(0.0)
        else:
            overlap = len(prev & names)
            turnovers.append(1.0 - overlap / cfg.top_k)
        prev = names
    return pd.Series(turnovers, index=by_date.index, name="turnover")


def _equity_curve(port_ret: pd.Series) -> pd.Series:
    return (1.0 + port_ret.fillna(0.0)).cumprod()


def top_k_hit_rate(
    df: pd.DataFrame,
    score_col: str,
    cfg: BacktestConfig,
    ret_col: str | None = None,
    ticker_col: str | None = None,
) -> float:
    """Fraction of overlap between predicted top-K and realized top-K each month."""
    data = df.copy()
    date_col = cfg.date_col
    ret_col = ret_col or cfg.ret_1m_col
    ticker_col = ticker_col or cfg.ticker_col
    k = cfg.top_k
    data[date_col] = pd.to_datetime(data[date_col])

    def _hit(x: pd.DataFrame) -> float:
        pred = _top_k_by_score(x, score_col, k)[ticker_col].tolist()
        actual = _top_k_by_score(x, ret_col, k)[ticker_col].tolist()
        if not pred:
            return np.nan
        return len(set(pred) & set(actual)) / k

    hit = data.groupby(date_col).apply(_hit)
    return float(hit.mean())


def _summarize_portfolio(
    port_ret: pd.Series,
    turnover: pd.Series | None = None,
    equity: pd.Series | None = None,
) -> dict:
    """Compute summary metrics for a monthly return series."""
    port_ret = port_ret.dropna()
    if equity is None:
        equity = _equity_curve(port_ret)
    max_dd = (equity / equity.cummax() - 1.0).min()
    vol = port_ret.std()
    sharpe = (port_ret.mean() / vol) * np.sqrt(12) if vol != 0 else np.nan
    cagr = (equity.iloc[-1] ** (12 / len(equity)) - 1.0) if len(equity) > 1 else np.nan
    mean_turnover = float(turnover.mean()) if turnover is not None else np.nan
    return {
        "mean_monthly_return": float(port_ret.mean()),
        "vol_monthly": float(vol),
        "sharpe": float(sharpe),
        "cagr": float(cagr),
        "max_drawdown": float(max_dd),
        "mean_turnover": mean_turnover,
    }


def _equal_weight_returns(df: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    """Equal-weight universe return (monthly mean of fwd_ret_1m)."""
    data = df.copy()
    data[cfg.date_col] = pd.to_datetime(data[cfg.date_col])
    if cfg.ret_1m_col not in data.columns:
        raise ValueError(f"Missing required column: {cfg.ret_1m_col}")
    port = data.groupby(cfg.date_col)[cfg.ret_1m_col].mean()
    port.name = "port_ret"
    return _apply_overlay(port, data, cfg)


def backtest_from_scores(
    df_scores: pd.DataFrame,
    score_cols: Dict[str, str],
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Monthly long-only top-K backtest for multiple score columns."""
    summary_rows = []
    artifacts: Dict[str, pd.DataFrame] = {}

    for model_name, score_col in score_cols.items():
        scores = df_scores.copy()
        scores[cfg.date_col] = pd.to_datetime(scores[cfg.date_col])

        top_k_rows = _top_k_panel(scores, score_col, cfg)
        port_ret = _portfolio_returns(scores, score_col, cfg, top_k_rows)
        equity = _equity_curve(port_ret)
        turnover = _turnover(top_k_rows, cfg)

        artifact = pd.DataFrame({"port_ret": port_ret, "equity": equity}).dropna(subset=["port_ret"])
        artifact = artifact.join(turnover, how="left")
        artifacts[model_name] = artifact

        summary = _summarize_portfolio(port_ret, turnover=turnover, equity=equity)
        summary_rows.append({
            "model": model_name,
            **summary,
            "hit_rate": top_k_hit_rate(scores, score_col, cfg),
        })

    summary = pd.DataFrame(summary_rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    return summary, artifacts


def equal_weight_benchmark(
    df: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return summary row and equity curve for equal-weight benchmark."""
    port_ret = _equal_weight_returns(df, cfg)
    equity = _equity_curve(port_ret)
    turnover = pd.Series(0.0, index=port_ret.index)

    summary = _summarize_portfolio(port_ret, turnover=turnover, equity=equity)
    rows = pd.DataFrame([{
        "model": "buy_hold_eqw",
        **summary,
        "hit_rate": np.nan,
    }])
    return rows, equity
