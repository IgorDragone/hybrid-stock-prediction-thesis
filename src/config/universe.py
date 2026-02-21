"""Helpers for loading and selecting deterministic ticker universes."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .io import load_yaml
from .paths import ROOT_DIR


def load_universe(path: Path | None = None) -> list[dict]:
    """
    Load the universe definition from YAML.

    Expected structure:
      sectors:
        - sector: Sector Name
          tickers: [AAA, BBB, ...]
    """
    if path is None:
        path = ROOT_DIR / "configs" / "universe.yaml"
    data = load_yaml(path)
    return data.get("sectors", [])


def _flatten(sectors: Iterable[dict]) -> list[str]:
    tickers: list[str] = []
    for entry in sectors:
        tickers.extend(entry.get("tickers", []))
    return tickers


def select_universe(
    n_tickers: int,
    mode: str = "balanced_by_sector",
    path: Path | None = None,
) -> list[str]:
    """
    Select a deterministic subset of tickers from the universe.

    Modes:
      - balanced_by_sector: round-robin selection across sectors (deterministic)
      - ordered: first N tickers in universe order
    """
    sectors = load_universe(path)
    all_tickers = _flatten(sectors)
    if n_tickers > len(all_tickers):
        raise ValueError(f"Requested {n_tickers} tickers, but universe has {len(all_tickers)}.")

    if mode == "ordered":
        return all_tickers[:n_tickers]

    if mode != "balanced_by_sector":
        raise ValueError(f"Unknown selection mode: {mode}")

    selection: list[str] = []
    idx = 0
    while len(selection) < n_tickers:
        for entry in sectors:
            tickers = entry.get("tickers", [])
            if idx < len(tickers):
                selection.append(tickers[idx])
                if len(selection) >= n_tickers:
                    break
        idx += 1

    return selection
