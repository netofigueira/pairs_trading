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
        incoming["open_time"] = pd.to_datetime(incoming["open_time"], utc=True, format="mixed")
        incoming["close_time"] = pd.to_datetime(incoming["close_time"], utc=True, format="mixed")
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["open_time", "close_time"])
            existing["open_time"] = pd.to_datetime(
                existing["open_time"], utc=True, format="mixed"
            )
            existing["close_time"] = pd.to_datetime(
                existing["close_time"], utc=True, format="mixed"
            )
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
        incoming["funding_time"] = pd.to_datetime(
            incoming["funding_time"], utc=True, format="mixed"
        )
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["funding_time"])
            existing["funding_time"] = pd.to_datetime(
                existing["funding_time"], utc=True, format="mixed"
            )
            incoming = pd.concat((existing, incoming), ignore_index=True)
        merged = incoming.drop_duplicates(subset="funding_time", keep="last").sort_values(
            "funding_time"
        )
        temporary = path.with_suffix(".tmp")
        merged.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
        return path

    def upsert_option_summaries(self, venue: str, currency: str, frame: pd.DataFrame) -> Path:
        """Atomically retain executable option-summary snapshots for research."""

        _require_columns(
            frame,
            {
                "snapshot_time",
                "instrument_name",
                "bid_price",
                "ask_price",
                "implied_volatility",
            },
        )
        path = self.root / "market" / venue / "options" / currency.upper() / "summaries.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = frame.copy()
        incoming["snapshot_time"] = pd.to_datetime(
            incoming["snapshot_time"], utc=True, format="mixed"
        )
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["snapshot_time"])
            existing["snapshot_time"] = pd.to_datetime(
                existing["snapshot_time"], utc=True, format="mixed"
            )
            incoming = pd.concat((existing, incoming), ignore_index=True)
        merged = incoming.drop_duplicates(
            subset=["snapshot_time", "instrument_name"], keep="last"
        ).sort_values(["snapshot_time", "instrument_name"])
        temporary = path.with_suffix(".tmp")
        merged.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
        return path

    def upsert_volatility_index(self, venue: str, currency: str, frame: pd.DataFrame) -> Path:
        """Atomically retain historical DVOL bars, deduplicated by timestamp."""

        _require_columns(frame, {"timestamp", "close"})
        path = self.root / "market" / venue / "volatility-index" / f"{currency.upper()}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = frame.copy()
        incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], utc=True, format="mixed")
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["timestamp"])
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, format="mixed")
            incoming = pd.concat((existing, incoming), ignore_index=True)
        merged = incoming.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
        temporary = path.with_suffix(".tmp")
        merged.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
        return path

    def write_option_chain_snapshot(self, venue: str, underlying: str, frame: pd.DataFrame) -> Path:
        """Persist one immutable, UTC-labelled historical option-chain snapshot."""

        _require_columns(frame, {"timestamp", "instrument", "bid_price", "ask_price", "source"})
        timestamps = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
        if timestamps.empty or timestamps.nunique() != 1:
            raise ValueError("an option-chain snapshot must contain exactly one timestamp")
        timestamp = timestamps.iloc[0]
        path = (
            self.root
            / "market"
            / venue
            / "options"
            / underlying.upper()
            / "chains"
            / f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        stored = frame.copy()
        stored["timestamp"] = timestamps
        stored.to_parquet(temporary, index=False)
        os.replace(temporary, path)
        return path

    def upsert_price_bars(
        self, venue: str, instrument: str, interval: str, frame: pd.DataFrame
    ) -> Path:
        """Atomically retain timestamped public OHLCV bars for a derivative instrument."""

        _require_columns(frame, {"timestamp", "close"})
        path = self.root / "market" / venue / "price-bars" / instrument / f"{interval}.csv.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        incoming = frame.copy()
        incoming["timestamp"] = pd.to_datetime(incoming["timestamp"], utc=True, format="mixed")
        if path.exists():
            existing = pd.read_csv(path, parse_dates=["timestamp"])
            existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, format="mixed")
            incoming = pd.concat((existing, incoming), ignore_index=True)
        merged = incoming.drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp")
        temporary = path.with_suffix(".tmp")
        merged.to_csv(temporary, index=False, compression="gzip")
        os.replace(temporary, path)
        return path


def _require_columns(frame: pd.DataFrame, required: set[str]) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"frame is missing required columns: {sorted(missing)}")
