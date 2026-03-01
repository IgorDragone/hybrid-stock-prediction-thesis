from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODELS_DIR


@dataclass(frozen=True)
class ModelBundle:
    model_id: str
    model_path: Path
    metrics_path: Path | None
    config_path: Path | None


def _registry_path(base_dir: Path | str) -> Path:
    return Path(base_dir) / "registry.json"


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"models": []}
    return json.loads(path.read_text())


def _save_registry(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def save_model_bundle(
    model_id: str,
    model: Any | None = None,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    base_dir: Path | str = MODELS_DIR,
) -> ModelBundle:
    """Persist model artifacts and update the registry."""
    base_dir = Path(base_dir)
    model_dir = base_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.pkl"
    metrics_path = model_dir / "metrics.json" if metrics is not None else None
    config_path = model_dir / "config.json" if config is not None else None

    if model is not None:
        joblib.dump(model, model_path)

    if metrics is not None:
        metrics_path.write_text(json.dumps(metrics, indent=2))

    if config is not None:
        config_path.write_text(json.dumps(config, indent=2))

    registry_path = _registry_path(base_dir)
    registry = _load_registry(registry_path)
    registry["models"] = [m for m in registry.get("models", []) if m.get("id") != model_id]
    registry["models"].append(
        {
            "id": model_id,
            "path": str(model_dir),
            "created_at": datetime.utcnow().isoformat(),
            "metrics": metrics or {},
            "config": config or {},
        }
    )
    _save_registry(registry_path, registry)

    return ModelBundle(
        model_id=model_id,
        model_path=model_path,
        metrics_path=metrics_path,
        config_path=config_path,
    )


def load_model_bundle(
    model_id: str,
    base_dir: Path | str = MODELS_DIR,
) -> tuple[Any | None, dict[str, Any], dict[str, Any]]:
    """Load model and metadata from disk."""
    try:
        import joblib
    except ModuleNotFoundError:
        joblib = None
    base_dir = Path(base_dir)
    model_dir = base_dir / model_id
    model_path = model_dir / "model.pkl"
    metrics_path = model_dir / "metrics.json"
    config_path = model_dir / "config.json"

    if joblib is not None and model_path.exists():
        model = joblib.load(model_path)
    else:
        model = None
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return model, metrics, config


def save_oos_scores(
    model_id: str,
    oos_df: Any,
    base_dir: Path | str = MODELS_DIR,
) -> Path:
    """Persist out-of-sample scores for later UI filtering."""
    base_dir = Path(base_dir)
    model_dir = base_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "oos_scores.parquet"
    oos_df.to_parquet(path, index=False)
    return path


def load_oos_scores(
    model_id: str,
    base_dir: Path | str = MODELS_DIR,
) -> Any | None:
    path = Path(base_dir) / model_id / "oos_scores.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def build_model_id(
    base: str,
    *,
    version: int = 1,
    use_macro: bool = False,
    funds_only: bool = False,
    tech_only: bool = False,
) -> str:
    parts = [base]
    if funds_only:
        parts.append("funds")
    if tech_only:
        parts.append("tech")
    if use_macro:
        parts.append("macro")
    parts.append(f"v{version}")
    return "_".join(parts)


def list_models(base_dir: Path | str = MODELS_DIR) -> list[str]:
    """Return model ids from the registry."""
    registry = _load_registry(_registry_path(Path(base_dir)))
    return [m.get("id") for m in registry.get("models", [])]
