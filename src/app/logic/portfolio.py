from __future__ import annotations

import json

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import DATA_DIR

PORTFOLIOS_PATH = DATA_DIR / "processed" / "app" / "portfolios.json"

@dataclass(frozen=True)
class Portfolio:
    name: str
    tickers: list[str]
    weights: dict[str, float]
    model_id: str | None = None
    rebalance_freq: str = "Monthly"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_portfolio(data: dict[str, Any]) -> None:
    if not data.get("name"):
        raise ValueError("Portfolio name is required.")
    tickers = data.get("tickers", [])
    if not tickers:
        raise ValueError("Portfolio must contain at least one ticker.")
    weights = data.get("weights", {})
    if not weights:
        raise ValueError("Portfolio weights are required.")
    missing = [t for t in tickers if t not in weights]
    if missing:
        raise ValueError(f"Missing weights for tickers: {missing}")


def save_portfolio(portfolio: Portfolio, path: Path | str) -> None:
    payload = portfolio.to_dict()
    _validate_portfolio(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_portfolio(path: Path | str) -> Portfolio:
    path = Path(path)
    payload = json.loads(path.read_text())
    _validate_portfolio(payload)
    return Portfolio(**payload)


def equal_weight(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        raise ValueError("Tickers list cannot be empty.")
    w = 1.0 / len(tickers)
    return {t: w for t in tickers}


def load_portfolios(path: Path | str = PORTFOLIOS_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Portfolios file must contain a list.")
    return payload


def save_portfolios(portfolios: list[dict[str, Any]], path: Path | str = PORTFOLIOS_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portfolios, indent=2))
