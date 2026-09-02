"""Contract selection and bid/ask accounting for the Tardis P1 plumbing gate."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


def select_atm_straddle(
    options: pd.DataFrame,
    *,
    underlying_mid: float,
    as_of: pd.Timestamp,
    min_dte: int = 7,
    max_dte: int = 30,
    target_dte: float = 14.0,
) -> pd.DataFrame:
    """Choose the expiry nearest a pre-declared DTE, then its ATM call/put pair.

    The horizon comes first so the tenor is a rule, not an artifact of which
    strike happened to sit closest to spot on a given day.
    """

    if underlying_mid <= 0 or min_dte < 0 or max_dte < min_dte:
        raise ValueError("invalid underlying price or DTE bounds")
    if not min_dte <= target_dte <= max_dte:
        raise ValueError("target_dte must lie within the DTE bounds")
    rows: list[dict[str, object]] = []
    for row in options.itertuples(index=False):
        parsed = _parse_option(str(row.symbol))
        if parsed is None:
            continue
        expiry, option_type, strike = parsed
        dte = (expiry - _utc(as_of)).total_seconds() / 86_400
        if min_dte <= dte <= max_dte:
            rows.append(
                {
                    "symbol": row.symbol,
                    "expiry": expiry,
                    "type": option_type,
                    "strike": strike,
                    "dte": dte,
                    "ask_price": row.ask_price,
                    "bid_price": row.bid_price,
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return candidates
    paired = candidates.pivot_table(
        index=["expiry", "strike", "dte"], columns="type", values="symbol", aggfunc="first"
    ).dropna()
    if paired.empty:
        return pd.DataFrame(columns=candidates.columns)
    chosen_expiry = min(
        {item[0]: item[2] for item in paired.index}.items(),
        key=lambda item: abs(item[1] - target_dte),
    )[0]
    expiry, strike, _ = min(
        (item for item in paired.index if item[0] == chosen_expiry),
        key=lambda item: abs(item[1] / underlying_mid - 1),
    )
    return (
        candidates.loc[(candidates["expiry"] == expiry) & (candidates["strike"] == strike)]
        .sort_values("type")
        .reset_index(drop=True)
    )


def _parse_option(symbol: str) -> tuple[pd.Timestamp, str, float] | None:
    pieces = symbol.split("-")
    if len(pieces) != 4 or pieces[3] not in {"C", "P"}:
        return None
    try:
        expiry = pd.Timestamp(datetime.strptime(pieces[1], "%d%b%y"), tz="UTC")
        expiry += pd.Timedelta(hours=8)
        return expiry, {"C": "call", "P": "put"}[pieces[3]], float(pieces[2])
    except ValueError:
        return None


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return timestamp.tz_convert("UTC")
