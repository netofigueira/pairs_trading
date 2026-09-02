"""Validation boundary for historical Deribit option-chain Parquet files.

The loader retains only quotes that could have been executed at the snapshot
instant. In particular, a zero price is an absent side, not a free option.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "timestamp",
    "underlying",
    "instrument",
    "strike",
    "expiry",
    "type",
    "mark_iv",
    "bid_price",
    "ask_price",
    "underlying_price",
    "delta",
    "source",
}
EXECUTABLE_SOURCES = frozenset({"live_ws", "live_rest"})


class VolarDataError(ValueError):
    """A historical option-chain file does not meet the declared contract."""


@dataclass(frozen=True)
class ValidatedOptionChain:
    quotes: pd.DataFrame
    source_rows: int
    executable_rows: int

    @property
    def rejected_rows(self) -> int:
        return self.source_rows - self.executable_rows


def load_executable_chain(path: Path | str) -> ValidatedOptionChain:
    """Load a Volar chain and remove non-live, stale, or unquoted observations."""

    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise VolarDataError(f"option-chain file is missing columns: {sorted(missing)}")
    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True, errors="raise")
    normalized["expiry"] = pd.to_datetime(normalized["expiry"], utc=True, errors="raise")
    for column in ("bid_price", "ask_price", "mark_iv", "underlying_price", "delta", "strike"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    live = normalized["source"].isin(EXECUTABLE_SOURCES)
    unexpired = normalized["expiry"] > normalized["timestamp"]
    quoted = normalized["bid_price"].gt(0) & normalized["ask_price"].gt(0)
    orderly = normalized["ask_price"].ge(normalized["bid_price"])
    valid = normalized[live & unexpired & quoted & orderly].copy()
    valid = valid.sort_values(["timestamp", "instrument"]).reset_index(drop=True)
    return ValidatedOptionChain(
        quotes=valid,
        source_rows=len(normalized),
        executable_rows=len(valid),
    )
