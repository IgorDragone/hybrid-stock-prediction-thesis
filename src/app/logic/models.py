from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import json

import numpy as np
import pandas as pd

from src.config import DATASETS_DIR, MODELS_DIR
from src.modeling.backtest import BacktestConfig, backtest_from_scores, equal_weight_returns, summarize_portfolio
from src.modeling.registry import load_model_bundle, load_oos_scores


def list_model_entries(base_dir: Path | str = MODELS_DIR) -> list[dict[str, Any]]:
    registry_path = Path(base_dir) / "registry.json"
    if not registry_path.exists():
        return []
    registry = json.loads(registry_path.read_text())
    return registry.get("models", [])


def model_metrics(model_id: str, base_dir: Path | str = MODELS_DIR) -> dict[str, Any]:
    path = Path(base_dir) / model_id / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def model_equity_curve(model_id: str, base_dir: Path | str = MODELS_DIR) -> dict[str, list[float]] | None:
    path = Path(base_dir) / model_id / "equity.csv"
    if not path.exists():
        return None
    rows = [line.strip().split(",") for line in path.read_text().strip().splitlines()]
    if not rows or rows[0] != ["date", "equity"]:
        return None
    dates = [r[0] for r in rows[1:] if len(r) == 2]
    values = [float(r[1]) for r in rows[1:] if len(r) == 2]
    return {"date": dates, "equity": values}


def _find_latest_model_ready() -> Path:
    candidates = list(DATASETS_DIR.glob("*/stages/panel_model_ready.parquet"))
    if not candidates:
        raise FileNotFoundError("No panel_model_ready.parquet found in data/processed/datasets.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_model_ready(tickers: list[str]) -> pd.DataFrame:
    path = _find_latest_model_ready()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if tickers:
        df = df[df["ticker"].isin(tickers)].copy()
    return df


def _load_stage_table(stage_name: str, tickers: list[str]) -> pd.DataFrame:
    candidates = list(DATASETS_DIR.glob(f"*/stages/{stage_name}.parquet"))
    if not candidates:
        return pd.DataFrame()
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    path = candidates[0]
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    if tickers:
        df = df[df["ticker"].isin(tickers)].copy()
    return df


def _infer_features(df: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    features = config.get("features")
    if features:
        return list(features)
    exclude = {
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
    return [c for c in df.columns if c not in exclude]


def compare_on_subset(
    model_ids: list[str],
    tickers: list[str],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    min_global_pct = 0.3
    df = _load_model_ready(tickers)
    if df.empty:
        raise ValueError("No data available for selected tickers.")

    n_tickers = df["ticker"].nunique()
    top_k = max(1, int(round(n_tickers * 0.4)))
    top_k = min(top_k, 10)
    cfg = BacktestConfig(top_k=top_k)

    summary_rows = []
    artifacts: Dict[str, pd.DataFrame] = {}

    df_scores = df.copy()
    score_cols: Dict[str, str] = {}
    ret_source = None

    for model_id in model_ids:
        model, _, config = load_model_bundle(model_id)
        model_type = config.get("type")

        if model_type == "benchmark" and model_id == "buy_hold_eqw":
            port_ret = equal_weight_returns(df_scores, cfg)
            metrics = summarize_portfolio(port_ret, turnover=None)
            metrics["model"] = model_id
            metrics["hit_rate"] = np.nan
            summary_rows.append(metrics)
            equity = (1.0 + port_ret.fillna(0.0)).cumprod()
            artifacts[model_id] = pd.DataFrame({"port_ret": port_ret, "equity": equity})
            continue

        if model_type == "baseline" and model_id == "baseline_mom":
            full_df = _load_model_ready([])
            if "mom12_pr" not in full_df.columns:
                raise ValueError("mom12_pr not available for baseline scoring.")
            full_df["score"] = full_df["mom12_pr"]
            full_df["date"] = pd.to_datetime(full_df["date"])
            if ret_source is None and "fwd_ret_1m" in full_df.columns:
                cols = ["date", "ticker", "fwd_ret_1m", "stress_index"]
                ret_source = full_df[[c for c in cols if c in full_df.columns]].copy()
            ranks = full_df.groupby("date")["score"].rank(ascending=False, method="first")
            counts = full_df.groupby("date")["score"].transform("count")
            full_df["rank_pct_global"] = np.where(
                counts > 1,
                (ranks - 1) / (counts - 1),
                0.0,
            )
            full_df["eligible"] = full_df["rank_pct_global"] <= min_global_pct
            base_cols = ["date", "ticker", "score", "eligible"]
            oos_trim = full_df[base_cols].copy()
            oos_trim = oos_trim[oos_trim["ticker"].isin(tickers)].copy()
            score_col = f"score_{model_id}"
            oos_trim = oos_trim.rename(columns={"score": score_col})
            oos_trim.loc[~oos_trim["eligible"], score_col] = np.nan
            oos_trim = oos_trim.drop(columns=["eligible"])
            if df_scores.empty:
                df_scores = oos_trim
            else:
                df_scores = df_scores.merge(
                    oos_trim,
                    on=[c for c in ["date", "ticker"] if c in df_scores.columns],
                    how="left",
                )
            score_cols[model_id] = score_col
            continue

        oos_df = load_oos_scores(model_id)
        if oos_df is None:
            raise ValueError(
                f"Missing OOS scores for {model_id}. Run notebook 03 to precompute."
            )
        if "score" not in oos_df.columns:
            fallback_cols = [
                c for c in oos_df.columns
                if c in {"score_baseline", "score_mom", "mom12_pr"} or c.startswith("score_")
            ]
            if len(fallback_cols) == 1:
                oos_df = oos_df.rename(columns={fallback_cols[0]: "score"})
            else:
                raise ValueError(
                    f"OOS scores for {model_id} must include a 'score' column."
                )
        oos_trim = oos_df.copy()
        oos_trim["date"] = pd.to_datetime(oos_trim["date"])
        ranks = oos_trim.groupby("date")["score"].rank(ascending=False, method="first")
        counts = oos_trim.groupby("date")["score"].transform("count")
        oos_trim["rank_pct_global"] = np.where(
            counts > 1,
            (ranks - 1) / (counts - 1),
            0.0,
        )
        oos_trim["eligible"] = oos_trim["rank_pct_global"] <= min_global_pct

        keep_cols = ["date", "ticker", "fwd_ret_1m", "stress_index", "score", "eligible"]
        oos_trim = oos_trim[[c for c in keep_cols if c in oos_trim.columns]].copy()
        if ret_source is None and "fwd_ret_1m" in oos_trim.columns:
            cols = ["date", "ticker", "fwd_ret_1m", "stress_index"]
            ret_source = oos_trim[[c for c in cols if c in oos_trim.columns]].copy()
        oos_trim = oos_trim[oos_trim["ticker"].isin(tickers)].copy()
        score_col = f"score_{model_id}"
        oos_trim = oos_trim.rename(columns={"score": score_col})
        if "eligible" in oos_trim.columns:
            oos_trim.loc[~oos_trim["eligible"], score_col] = np.nan
            oos_trim = oos_trim.drop(columns=["eligible"])
        if df_scores.empty:
            df_scores = oos_trim
        else:
            df_scores = df_scores.merge(
                oos_trim,
                on=[c for c in ["date", "ticker"] if c in df_scores.columns],
                how="left",
            )
        score_cols[model_id] = score_col

    if "fwd_ret_1m" not in df_scores.columns:
        if ret_source is not None:
            df_scores = df_scores.merge(
                ret_source,
                on=[c for c in ["date", "ticker"] if c in df_scores.columns],
                how="left",
            )
        if "fwd_ret_1m" not in df_scores.columns:
            eom_df = _load_stage_table("panel_eom", tickers)
            if not eom_df.empty and "fwd_ret_1m" in eom_df.columns:
                cols = ["date", "ticker", "fwd_ret_1m", "stress_index"]
                eom_df = eom_df[[c for c in cols if c in eom_df.columns]].copy()
                df_scores = df_scores.merge(
                    eom_df,
                    on=[c for c in ["date", "ticker"] if c in df_scores.columns],
                    how="left",
                )
        if "fwd_ret_1m" not in df_scores.columns:
            raise ValueError(
                "Missing required column: fwd_ret_1m. "
                "Recompute OOS scores with fwd_ret_1m included."
            )

    if score_cols:
        summary, artifacts_scores = backtest_from_scores(df_scores, score_cols, cfg)
        summary_rows.extend(summary.to_dict(orient="records"))
        for model_id, art in artifacts_scores.items():
            if "port_ret" in art.columns:
                art = art.dropna(subset=["port_ret"]).copy()
                art["equity"] = (1.0 + art["port_ret"]).cumprod()
            artifacts[model_id] = art

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty and "model" in summary_df.columns:
        summary_df = summary_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    return summary_df, artifacts
