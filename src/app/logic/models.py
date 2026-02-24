from __future__ import annotations

from pathlib import Path
from typing import Any

import json

from src.config import MODELS_DIR


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
