from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DATASETS_DIR, load_universe
from src.modeling.registry import load_model_bundle

TICKER_COMPANY = {
    "AAPL": "Apple Inc.",
    "ABBV": "AbbVie Inc.",
    "AEP": "American Electric Power Company, Inc.",
    "AMT": "American Tower Corporation",
    "AMZN": "Amazon.com, Inc.",
    "AXP": "American Express Company",
    "BA": "The Boeing Company",
    "BAC": "Bank of America Corporation",
    "CAT": "Caterpillar Inc.",
    "COP": "ConocoPhillips",
    "COST": "Costco Wholesale Corporation",
    "CVX": "Chevron Corporation",
    "D": "Dominion Energy, Inc.",
    "DUK": "Duke Energy Corporation",
    "EQIX": "Equinix, Inc.",
    "GE": "GE Aerospace",
    "GOOGL": "Alphabet Inc.",
    "GS": "The Goldman Sachs Group, Inc.",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "KO": "The Coca-Cola Company",
    "MCD": "McDonald's Corporation",
    "META": "Meta Platforms, Inc.",
    "MMM": "3M Company",
    "MRK": "Merck & Co., Inc.",
    "MS": "Morgan Stanley",
    "MSFT": "Microsoft Corporation",
    "NEE": "NextEra Energy, Inc.",
    "NKE": "NIKE, Inc.",
    "NVDA": "NVIDIA Corporation",
    "PEP": "PepsiCo, Inc.",
    "PFE": "Pfizer Inc.",
    "PG": "The Procter & Gamble Company",
    "PLD": "Prologis, Inc.",
    "PSA": "Public Storage",
    "SBUX": "Starbucks Corporation",
    "SHEL": "Shell plc",
    "SO": "The Southern Company",
    "SPG": "Simon Property Group, Inc.",
    "TSLA": "Tesla, Inc.",
    "TTE": "TotalEnergies SE",
    "UNH": "UnitedHealth Group Incorporated",
    "UNP": "Union Pacific Corporation",
    "WMT": "Walmart Inc.",
    "XOM": "Exxon Mobil Corporation",
}


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


def _recommendation_level(rank_pct_global: float) -> str:
    if rank_pct_global <= 0.10:
        return "Very High"
    if rank_pct_global <= 0.30:
        return "High"
    if rank_pct_global < 0.50:
        return "Medium (Upper)"
    if rank_pct_global < 0.70:
        return "Medium (Lower)"
    if rank_pct_global < 0.90:
        return "Low"
    return "Very Low"


def _ticker_sector_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    try:
        universe = load_universe()
        for entry in universe:
            sector = entry.get("sector", "Unknown")
            for ticker in entry.get("tickers", []):
                mapping[str(ticker)] = str(sector)
    except Exception:  # noqa: BLE001
        return {}
    return mapping


def score_watchlist(
    model_id: str,
    tickers: list[str],
    top_k: int | None = None,
) -> dict[str, Any]:
    buy_threshold = 0.30
    sell_threshold = 0.70

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

    snap = snap_full[snap_full["ticker"].isin(snap_subset["ticker"])].copy()
    snap = snap.sort_values("score", ascending=False).copy()
    snap["company"] = snap["ticker"].map(TICKER_COMPANY).fillna(snap["ticker"])
    sector_map = _ticker_sector_map()
    snap["sector"] = snap["ticker"].map(sector_map).fillna("Unknown")
    snap["rank"] = np.arange(1, len(snap) + 1)
    if len(snap) > 1:
        snap["rank_pct"] = (snap["rank"] - 1) / (len(snap) - 1)
    else:
        snap["rank_pct"] = 0.0
    snap["action"] = "HOLD ⏸️"
    snap.loc[snap["rank_pct_global"] <= buy_threshold, "action"] = "BUY ✅"
    snap.loc[snap["rank_pct_global"] >= sell_threshold, "action"] = "SELL ⛔"
    snap["recommendation_level"] = snap["rank_pct_global"].apply(_recommendation_level)

    exposure = 1.0
    if "stress_index" in snap.columns:
        stress = float(snap["stress_index"].iloc[0])
        exposure = 0.6 if stress > 0.5 else 1.0
    else:
        stress = None

    buy_count = int((snap["action"] == "BUY ✅").sum())

    universe_size = len(snap_full)
    recommendations = snap[
        [
            "rank",
            "ticker",
            "company",
            "sector",
            "action",
            "recommendation_level",
            "rank_global",
            "rank_pct",
            "rank_pct_global",
        ]
    ].copy()
    recommendations = recommendations.set_index("rank")
    return {
        "date": latest_date,
        "exposure": exposure,
        "stress_index": stress,
        "buy_count": buy_count,
        "universe_size": universe_size,
        "used_fallback": used_fallback,
        "recommendations": recommendations,
        "metrics": metrics,
    }
