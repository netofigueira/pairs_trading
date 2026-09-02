"""Coverage gates for an executable historical option-quote study.

The purpose of this module is to verify that a trade could have been entered
and exited on observed two-sided quotes for the *same* contract.  It deliberately
does not turn a mark-IV series into a synthetic fill.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.volar import load_executable_chain


def load_quote_snapshots(paths: list[Path]) -> pd.DataFrame:
    """Load validated chain snapshots, rejecting files with mixed timestamps."""

    if not paths:
        raise ValueError("at least one chain snapshot is required")
    snapshots: list[pd.DataFrame] = []
    for path in sorted(paths):
        quotes = load_executable_chain(path).quotes
        timestamps = quotes["timestamp"].drop_duplicates()
        if len(timestamps) != 1:
            raise ValueError(f"{path} does not contain exactly one quote timestamp")
        snapshots.append(quotes)
    return pd.concat(snapshots, ignore_index=True).sort_values(
        ["timestamp", "instrument"]
    ).reset_index(drop=True)


def round_trip_coverage(
    quotes: pd.DataFrame,
    *,
    horizon: pd.Timedelta = pd.Timedelta(days=7),
    tolerance: pd.Timedelta = pd.Timedelta(hours=1),
) -> pd.DataFrame:
    """Match entry quotes with a near-target later snapshot of the same option.

    A row confirms only data availability: a long would pay ``entry_ask`` then
    receive ``exit_bid``.  No P&L claim is made because hedge and risk rules
    belong to the later executable-strategy specification.
    """

    required = {"timestamp", "instrument", "bid_price", "ask_price"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"quotes are missing columns: {sorted(missing)}")
    if horizon <= pd.Timedelta(0) or tolerance < pd.Timedelta(0):
        raise ValueError("horizon must be positive and tolerance cannot be negative")
    frame = quotes.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    timestamps = sorted(frame["timestamp"].unique())
    matches: list[pd.DataFrame] = []
    for entry_time in timestamps:
        target = entry_time + horizon
        candidates = [time for time in timestamps if abs(time - target) <= tolerance]
        if not candidates:
            continue
        exit_time = min(candidates, key=lambda time: abs(time - target))
        if exit_time <= entry_time:
            continue
        entry = frame.loc[frame["timestamp"] == entry_time, ["instrument", "ask_price"]]
        exit = frame.loc[frame["timestamp"] == exit_time, ["instrument", "bid_price"]]
        paired = entry.merge(exit, on="instrument", how="inner", validate="one_to_one")
        if paired.empty:
            continue
        paired.insert(0, "entry_time", entry_time)
        paired.insert(1, "exit_time", exit_time)
        paired = paired.rename(columns={"ask_price": "entry_ask", "bid_price": "exit_bid"})
        matches.append(paired)
    if not matches:
        return pd.DataFrame(
            columns=["entry_time", "exit_time", "instrument", "entry_ask", "exit_bid"]
        )
    return pd.concat(matches, ignore_index=True)


def coverage_summary(quotes: pd.DataFrame, matches: pd.DataFrame) -> dict[str, float | int]:
    """Summarize sampling cadence and exact-contract round-trip availability."""

    timestamps = pd.Series(pd.to_datetime(quotes["timestamp"].unique(), utc=True)).sort_values()
    gaps = timestamps.diff().dropna().dt.total_seconds() / 3_600
    entry_times = matches["entry_time"].nunique() if not matches.empty else 0
    return {
        "snapshots": int(len(timestamps)),
        "executable_quotes": int(len(quotes)),
        "unique_contracts": int(quotes["instrument"].nunique()),
        "coverage_days": float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 86_400),
        "median_snapshot_gap_hours": float(gaps.median()) if not gaps.empty else 0.0,
        "max_snapshot_gap_hours": float(gaps.max()) if not gaps.empty else 0.0,
        "entry_snapshots_with_exit": int(entry_times),
        "exact_contract_round_trips": int(len(matches)),
    }
