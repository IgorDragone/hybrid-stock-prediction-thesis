from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATASETS_DIR
from src.modeling.registry import load_model_bundle


def _find_latest_model_ready() -> Path:
    candidates = list(DATASETS_DIR.glob("*/stages/panel_model_ready.parquet"))
    if not candidates:
        raise FileNotFoundError("No panel_model_ready.parquet found in data/processed/datasets.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_snapshot(tickers: list[str]) -> tuple[pd.DataFrame, pd.Timestamp]:
    path = _find_latest_model_ready()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    snap = df[df["date"] == latest_date].copy()
    if tickers:
        snap = snap[snap["ticker"].isin(tickers)].copy()
    return snap, latest_date


def score_portfolio(
    model_id: str,
    tickers: list[str],
    cash: float,
    top_k: int | None = None,
) -> dict[str, Any]:
    min_global_pct = 0.3

    snap_full, latest_date = _load_snapshot([])
    if snap_full.empty:
        raise ValueError("No data available for scoring.")
    snap_subset = snap_full
    if tickers:
        snap_subset = snap_full[snap_full["ticker"].isin(tickers)].copy()
    if snap_subset.empty:
        raise ValueError("No data available for selected tickers on latest date.")

    model, metrics, config = load_model_bundle(model_id)
    model_type = config.get("type")
    features = config.get("features") or [
        c for c in snap_full.columns
        if c not in {
            "date",
            "ticker",
            "target_3m",
            "target_1m",
            "fwd_ret_1m",
            "fwd_ret_3m",
            "fwd_ret_6m",
            "macro_regime",
            "stress_index",
            "slope_10y2y",
        }
    ]

    used_fallback = False
    if model_type == "baseline":
        if "mom12_pr" not in snap_full.columns:
            raise ValueError("mom12_pr not available for baseline scoring.")
        snap_full["score"] = snap_full["mom12_pr"]
    elif model_type == "benchmark":
        raise ValueError("Benchmark entries cannot be used for portfolio scoring.")
    elif model is None:
        used_fallback = True
        # Fallback: momentum score when the trained model cannot be loaded.
        if "mom12_pr" not in snap_full.columns:
            raise ValueError("mom12_pr not available for fallback scoring.")
        snap_full["score"] = snap_full["mom12_pr"]
    else:
        snap_full["score"] = model.predict(snap_full[features])

    snap_full = snap_full.sort_values("score", ascending=False).copy()
    snap_full["rank_global"] = np.arange(1, len(snap_full) + 1)
    if len(snap_full) > 1:
        snap_full["rank_pct_global"] = (snap_full["rank_global"] - 1) / (len(snap_full) - 1)
    else:
        snap_full["rank_pct_global"] = 0.0
    snap_full["eligible"] = snap_full["rank_pct_global"] <= min_global_pct

    snap = snap_full[snap_full["ticker"].isin(snap_subset["ticker"])].copy()
    snap = snap.sort_values("score", ascending=False).copy()
    snap["rank"] = np.arange(1, len(snap) + 1)
    if len(snap) > 1:
        snap["rank_pct"] = (snap["rank"] - 1) / (len(snap) - 1)
    else:
        snap["rank_pct"] = 0.0
    if top_k is None:
        top_k = max(3, int(round(len(snap) * 0.33)))
    top_k = min(top_k, len(snap), 5)
    eligible = snap[snap["eligible"]].copy()
    top_k = min(top_k, len(eligible))
    if top_k > 0:
        buy_tickers = eligible.head(top_k)["ticker"]
        snap["action"] = np.where(snap["ticker"].isin(buy_tickers), "BUY ✅", "SELL ⛔")
    else:
        snap["action"] = "SELL ⛔"

    exposure = 1.0
    if "stress_index" in snap.columns:
        stress = float(snap["stress_index"].iloc[0])
        exposure = 0.6 if stress > 0.5 else 1.0
    else:
        stress = None

    investable_cash = cash * exposure if top_k > 0 else 0.0
    cash_left = cash - investable_cash
    snap["allocation_eur"] = 0.0
    if top_k > 0:
        per_buy = investable_cash / top_k
        snap.loc[snap["action"] == "BUY ✅", "allocation_eur"] = per_buy

    recommendations = snap[
        ["rank", "ticker", "action", "allocation_eur", "score", "rank_pct", "rank_pct_global"]
    ].copy()
    recommendations = recommendations.set_index("rank")
    return {
        "date": latest_date,
        "exposure": exposure,
        "stress_index": stress,
        "cash_left": cash_left,
        "used_fallback": used_fallback,
        "recommendations": recommendations,
        "metrics": metrics,
    }
