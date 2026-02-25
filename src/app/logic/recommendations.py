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
    snap, latest_date = _load_snapshot(tickers)
    if snap.empty:
        raise ValueError("No data available for selected tickers on latest date.")

    model, metrics, config = load_model_bundle(model_id)
    features = config.get("features") or [
        c for c in snap.columns
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

    if model is None:
        # Baseline: momentum score
        if "mom12_pr" not in snap.columns:
            raise ValueError("mom12_pr not available for baseline scoring.")
        snap["score"] = snap["mom12_pr"]
    else:
        snap["score"] = model.predict(snap[features])

    snap = snap.sort_values("score", ascending=False).copy()
    snap["rank"] = np.arange(1, len(snap) + 1)
    if top_k is None:
        top_k = max(1, int(round(len(snap) * 0.4)))
    top_k = min(top_k, len(snap), 10)
    snap["action"] = np.where(snap["rank"] <= top_k, "buy", "sell")

    exposure = 1.0
    if "stress_index" in snap.columns:
        stress = float(snap["stress_index"].iloc[0])
        exposure = 0.6 if stress > 0.5 else 1.0
    else:
        stress = None

    investable_cash = cash * exposure
    cash_left = cash - investable_cash
    allocation = np.zeros(len(snap))
    if top_k > 0:
        allocation[:top_k] = investable_cash / top_k
    snap["allocation_eur"] = allocation

    recommendations = snap[["rank", "ticker", "action", "allocation_eur"]].copy()
    recommendations = recommendations.set_index("rank")
    return {
        "date": latest_date,
        "exposure": exposure,
        "stress_index": stress,
        "cash_left": cash_left,
        "recommendations": recommendations,
        "metrics": metrics,
    }
