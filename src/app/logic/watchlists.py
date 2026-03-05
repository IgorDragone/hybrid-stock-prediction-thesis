from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import DATA_DIR

WATCHLISTS_PATH = DATA_DIR / "processed" / "app" / "watchlists.json"
LEGACY_PORTFOLIOS_PATH = DATA_DIR / "processed" / "app" / "portfolios.json"


def load_watchlists(path: Path | str = WATCHLISTS_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists() and path == WATCHLISTS_PATH and LEGACY_PORTFOLIOS_PATH.exists():
        path = LEGACY_PORTFOLIOS_PATH
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("Watchlists file must contain a list.")
    return payload


def save_watchlists(
    watchlists: list[dict[str, Any]],
    path: Path | str = WATCHLISTS_PATH,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(watchlists, indent=2))
