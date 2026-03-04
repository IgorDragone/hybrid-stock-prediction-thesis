from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return a dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str | Path) -> None:
    """Save a dict as YAML."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def save_json(data: dict, path: str | Path) -> None:
    """Save a dict as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def save_run_snapshot(cfg: dict, dataset_dir: str | Path) -> Path:
    """
    Save the run spec (input config) into the dataset folder as config.yaml.
    This should be called after a successful build.
    """
    out_path = Path(dataset_dir) / "config.yaml"
    save_yaml(cfg, out_path)
    return out_path


def summarize_stage(
    path: str | Path,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> dict:
    """Summarize a saved stage file for the dataset manifest."""
    stage_path = Path(path)
    df = pd.read_parquet(stage_path)

    summary = {
        "path": str(stage_path),
        "rows": int(len(df)),
        "n_columns": int(len(df.columns)),
    }

    if ticker_col in df.columns:
        summary["n_tickers"] = int(df[ticker_col].nunique())

    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col])
        summary["date_min"] = str(dates.min().date())
        summary["date_max"] = str(dates.max().date())

    return summary


def build_pipeline_manifest(
    dataset_id: str,
    stage_paths: dict[str, str | Path],
    *,
    final_stage: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> dict:
    """Create a manifest that summarizes every saved pipeline stage."""
    if final_stage not in stage_paths:
        raise KeyError(f"final_stage '{final_stage}' not found in stage_paths")

    stage_summaries = {
        stage_name: summarize_stage(path, date_col=date_col, ticker_col=ticker_col)
        for stage_name, path in stage_paths.items()
    }

    return {
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": list(stage_paths.keys()),
        "final_stage": final_stage,
        "stage_summaries": stage_summaries,
    }


def save_manifest(manifest: dict, dataset_dir: str | Path) -> Path:
    """Save manifest.json into the dataset folder."""
    out_path = Path(dataset_dir) / "manifest.json"
    save_json(manifest, out_path)
    return out_path
