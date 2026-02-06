from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    date_col: str = "date"
    ticker_col: str = "ticker"
    ret_1d_col: str = "ret_1d"
    rebalance_step: int = 21      # ~monthly
    q: float = 0.3                # top/bottom quantile
    lag_weights_by_1d: bool = True


def _rebalance_dates(df: pd.DataFrame, date_col: str, step: int) -> list[pd.Timestamp]:
    dates = df[date_col].drop_duplicates().sort_values().to_list()
    return dates[::step]


def _weights_for_date(x: pd.DataFrame, score_col: str, q: float) -> pd.Series:
    q_low = x[score_col].quantile(q)
    q_high = x[score_col].quantile(1 - q)
    sig = np.where(x[score_col] >= q_high, 1, np.where(x[score_col] <= q_low, -1, 0))

    w = np.zeros(len(x), dtype=float)
    n_long = (sig == 1).sum()
    n_short = (sig == -1).sum()

    if n_long > 0:
        w[sig == 1] = 1.0 / n_long
    # if long-only, we don't short anything
    if n_short > 0:
        w[sig == -1] = -1.0 / n_short

    return pd.Series(w, index=x.index)


def build_monthly_weights(df: pd.DataFrame, score_col: str, cfg: BacktestConfig) -> pd.DataFrame:
    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])
    d = d.sort_values([cfg.date_col, cfg.ticker_col])

    for c in [cfg.date_col, cfg.ticker_col, cfg.ret_1d_col, score_col]:
        if c not in d.columns:
            raise ValueError(f"Missing required column: {c}")

    reb_dates = _rebalance_dates(d, cfg.date_col, cfg.rebalance_step)
    reb_set = set(reb_dates)

    d["weight"] = np.nan
    mask = d[cfg.date_col].isin(reb_set)

    d.loc[mask, "weight"] = (
        d.loc[mask]
         .groupby(cfg.date_col, group_keys=False)
         .apply(lambda x: _weights_for_date(x, score_col, cfg.q))
    )

    d["weight"] = d.groupby(cfg.ticker_col)["weight"].ffill().fillna(0.0)
    return d


def portfolio_returns_from_weights(df_w: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    d = df_w.sort_values([cfg.ticker_col, cfg.date_col]).copy()

    if cfg.lag_weights_by_1d:
        d["w_eff"] = d.groupby(cfg.ticker_col)["weight"].shift(1).fillna(0.0)
    else:
        d["w_eff"] = d["weight"]

    port = d.groupby(cfg.date_col).apply(lambda x: (x["w_eff"] * x[cfg.ret_1d_col]).sum())
    port.name = "port_ret"
    return port


def equity_curve(port_ret: pd.Series) -> pd.Series:
    return (1.0 + port_ret.fillna(0.0)).cumprod()


def turnover(df_w: pd.DataFrame, cfg: BacktestConfig) -> pd.Series:
    d = df_w.sort_values([cfg.ticker_col, cfg.date_col]).copy()
    chg = d.groupby(cfg.ticker_col)["weight"].diff().abs().fillna(0.0)
    return chg.groupby(d[cfg.date_col]).mean()


def backtest_from_scores(
    df_scores: pd.DataFrame,
    score_cols: Dict[str, str],
    cfg: BacktestConfig,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Monthly-rebalanced long-short backtest for multiple score columns."""
    summary_rows = []
    artifacts: Dict[str, pd.DataFrame] = {}

    for model_name, score_col in score_cols.items():
        df_w = build_monthly_weights(df_scores, score_col, cfg)
        pr = portfolio_returns_from_weights(df_w, cfg)
        eq = equity_curve(pr)
        to = turnover(df_w, cfg)

        art = pd.DataFrame({"port_ret": pr, "equity": eq, "turnover": to}).dropna(subset=["port_ret"])
        artifacts[model_name] = art

        r = art["port_ret"]
        e = art["equity"]
        max_dd = (e / e.cummax() - 1.0).min()

        vol = r.std()
        sharpe_like = (r.mean() / vol) if vol != 0 else np.nan

        summary_rows.append({
            "model": model_name,
            "mean_daily_return": float(r.mean()),
            "vol_daily": float(vol),
            "sharpe_like": float(sharpe_like),
            "hit_ratio": float((r > 0).mean()),
            "max_drawdown": float(max_dd),
            "mean_turnover": float(art["turnover"].mean()),
        })

    summary = pd.DataFrame(summary_rows).sort_values("sharpe_like", ascending=False).reset_index(drop=True)
    return summary, artifacts
