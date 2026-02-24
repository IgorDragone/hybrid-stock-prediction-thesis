from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "processed" / "datasets"
MODELS_DIR = DATA_DIR / "processed" / "models"
RESULTS_DIR = ROOT_DIR / "results"
