"""Reconstruct executable top-of-book snapshots from Tardis quote updates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "exchange",
    "symbol",
    "timestamp",
    "local_timestamp",
    "ask_amount",
    "ask_price",
    "bid_price",
    "bid_amount",
}


def reconstruct_top_of_book(
    path: Path | str,
    *,
    as_of: pd.Timestamp | None = None,
    chunk_rows: int = 250_000,
    max_age: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Return the final valid quote state of each instrument in one raw CSV.GZ.

    Tardis quote messages are updates: an event can change only bid *or* ask.
    The function therefore carries each side forward per symbol and rejects
    zero, absent or crossed books. It never turns a missing side into a fill.
    """

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if max_age is not None and max_age < pd.Timedelta(0):
        raise ValueError("max_age cannot be negative")
    cutoff_us = None if as_of is None else int(_utc(as_of).timestamp() * 1_000_000)
    latest: dict[str, pd.DataFrame] = {}
    for chunk in pd.read_csv(path, chunksize=chunk_rows):
        missing = REQUIRED_COLUMNS.difference(chunk.columns)
        if missing:
            raise ValueError(f"Tardis quote file is missing columns: {sorted(missing)}")
        if cutoff_us is not None:
            chunk = chunk.loc[chunk["timestamp"] <= cutoff_us]
            if chunk.empty:
                continue
        latest["events"] = _latest_per_symbol(
            latest.get("events"), chunk[["symbol", "timestamp", "local_timestamp"]]
        )
        for field in ("bid_price", "bid_amount", "ask_price", "ask_amount"):
            updates = chunk.loc[chunk[field].notna(), ["symbol", field]]
            latest[field] = _latest_per_symbol(latest.get(field), updates)
    if "events" not in latest:
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    frame = latest["events"]
    for field in ("bid_price", "bid_amount", "ask_price", "ask_amount"):
        frame = frame.merge(latest[field], on="symbol", how="left", validate="one_to_one")
    if frame.empty:
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="us", utc=True)
    frame["local_timestamp"] = pd.to_datetime(frame["local_timestamp"], unit="us", utc=True)
    for field in ("bid_price", "bid_amount", "ask_price", "ask_amount"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    valid = (
        frame["bid_price"].gt(0)
        & frame["ask_price"].gt(0)
        & frame["bid_amount"].gt(0)
        & frame["ask_amount"].gt(0)
        & frame["ask_price"].ge(frame["bid_price"])
    )
    if max_age is not None:
        reference = _utc(as_of) if as_of is not None else frame["timestamp"].max()
        valid &= frame["timestamp"].ge(reference - max_age)
    return frame.loc[valid].sort_values("symbol").reset_index(drop=True)


def _latest_per_symbol(existing: pd.DataFrame | None, updates: pd.DataFrame) -> pd.DataFrame:
    """Merge a batch of updates, retaining only the last row for each symbol."""

    combined = updates if existing is None else pd.concat((existing, updates), ignore_index=True)
    return combined.drop_duplicates("symbol", keep="last")


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return timestamp.tz_convert("UTC")
