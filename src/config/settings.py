from __future__ import annotations

from pathlib import Path
import os
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CONFIGS_DIR = ROOT_DIR / "configs"
DATASETS_DIR = DATA_DIR / "processed" / "datasets"
RESULTS_DIR = ROOT_DIR / "results"


def load_config(name: str) -> dict:
    """Load a run spec YAML from configs/."""
    cfg_path = CONFIGS_DIR / name
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_dir(cfg: dict) -> Path:
    """Resolve dataset directory for a run spec."""
    dataset_id = cfg.get("dataset_id")
    if not dataset_id:
        raise ValueError("Config is missing dataset_id")
    return DATASETS_DIR / dataset_id


def get_env(name: str, default: str | None = None) -> str | None:
    """Helper to fetch env vars with an optional default."""
    return os.getenv(name, default)
