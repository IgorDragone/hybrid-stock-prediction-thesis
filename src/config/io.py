from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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


def build_manifest(
    cfg: dict,
    *,
    rows: int | None = None,
    n_tickers: int | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
    stages: Iterable[str] | None = None,
    notes: str | None = None,
    git_commit: str | None = None,
) -> dict:
    """Create a manifest payload for a dataset build."""
    return {
        "dataset_id": cfg.get("dataset_id"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "rows": rows,
        "n_tickers": n_tickers,
        "date_min": date_min,
        "date_max": date_max,
        "stages": list(stages) if stages else [],
        "notes": notes,
    }


def build_manifest_from_df(
    cfg: dict,
    df,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    stages: Iterable[str] | None = None,
    notes: str | None = None,
    git_commit: str | None = None,
) -> dict:
    """Convenience wrapper to build a manifest using a DataFrame."""
    date_min = None
    date_max = None
    n_tickers = None
    if date_col in df.columns:
        date_min = str(df[date_col].min().date())
        date_max = str(df[date_col].max().date())
    if ticker_col in df.columns:
        n_tickers = df[ticker_col].nunique()
    return build_manifest(
        cfg,
        rows=len(df),
        n_tickers=n_tickers,
        date_min=date_min,
        date_max=date_max,
        stages=stages,
        notes=notes,
        git_commit=git_commit,
    )


def save_manifest(manifest: dict, dataset_dir: str | Path) -> Path:
    """Save manifest.json into the dataset folder."""
    out_path = Path(dataset_dir) / "manifest.json"
    save_json(manifest, out_path)
    return out_path
