"""Small, append-safe local data lake for research inputs.

Raw market data is intentionally excluded from Git. A single canonical CSV.GZ
per instrument/interval makes every backtest point to a tangible data snapshot
without adding a database dependency in the MVP.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class LocalDataLake:
    def __init__(self, root: Path | str = "data") -> None:
        self.root = Path(root)

    def upsert_klines(self, venue: str, symbol: str, interval: str, frame: pd.DataFrame) -> Path:
        """Atomically merge a kline frame, deduplicated by open_time."""

        _require_columns(frame, {"open_time", "close_time", "close"})
        path = self.root / "market" / venue / "klines" / symbol.upper() / f"{interval}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = frame.copy()
        incoming["open_time"] = pd.to_datetime(incoming["open_time"], utc=True)
        incoming["close_time"] = pd.to_datetime(incoming["close_time"], utc=True)
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["open_time", "close_time"])
            existing["open_time"] = pd.to_datetime(existing["open_time"], utc=True)
            existing["close_time"] = pd.to_datetime(existing["close_time"], utc=True)
            incoming = pd.concat((existing, incoming), ignore_index=True)
        merged = incoming.drop_duplicates(subset="open_time", keep="last").sort_values("open_time")
        temporary = path.with_suffix(".tmp")
        merged.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
        return path

    def upsert_funding(self, venue: str, symbol: str, frame: pd.DataFrame) -> Path:
        """Atomically merge funding events, deduplicated by funding_time."""

        _require_columns(frame, {"funding_time", "funding_rate"})
        path = self.root / "market" / venue / "funding" / f"{symbol.upper()}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = frame.copy()
        incoming["funding_time"] = pd.to_datetime(incoming["funding_time"], utc=True)
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["funding_time"])
            existing["funding_time"] = pd.to_datetime(existing["funding_time"], utc=True)
            incoming = pd.concat((existing, incoming), ignore_index=True)
        merged = incoming.drop_duplicates(subset="funding_time", keep="last").sort_values(
            "funding_time"
        )
        temporary = path.with_suffix(".tmp")
        merged.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
        return path


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
