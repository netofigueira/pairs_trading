"""Contract selection and bid/ask accounting for the Tardis P1 plumbing gate."""

from __future__ import annotations

import math
from datetime import datetime

import pandas as pd

from quant_pairs.inverse_options import _normal_cdf, implied_volatility


def select_strangle_by_delta(
    options: pd.DataFrame,
    *,
    forward: float,
    as_of: pd.Timestamp,
    expiry: pd.Timestamp,
    target_delta: float = 0.25,
    min_contracts: float = 0.0,
) -> pd.DataFrame:
    """Pick the OTM call/put pair closest to a target Black-76 forward delta.

    Mid IVs are inverted per strike against the given forward; strikes whose
    inversion fails or whose displayed size is below ``min_contracts`` are
    skipped.  The call targets ``+target_delta`` (N(d1)) and the put
    ``-target_delta`` (N(d1) - 1).  Returns an empty frame when either side
    has no candidate.
    """

    if forward <= 0 or not 0 < target_delta < 0.5:
        raise ValueError("forward must be positive and target_delta in (0, 0.5)")
    expiry_utc = _utc(expiry)
    time_years = (expiry_utc - _utc(as_of)).total_seconds() / (365 * 86_400)
    if time_years <= 0:
        raise ValueError("expiry must be after as_of")
    candidates: list[dict[str, object]] = []
    for row in options.itertuples(index=False):
        parsed = _parse_option(str(row.symbol))
        if parsed is None or parsed[0] != expiry_utc:
            continue
        option_type, strike = parsed[1], parsed[2]
        bid = float(row.bid_price)
        ask = float(row.ask_price)
        if min(float(row.bid_amount), float(row.ask_amount)) < min_contracts:
            continue
        mid = (bid + ask) / 2
        try:
            mid_iv = implied_volatility(
                option_type,
                price_btc=mid,
                forward=forward,
                strike=strike,
                time_years=time_years,
            )
        except ValueError:
            continue
        d1 = (math.log(forward / strike) + 0.5 * mid_iv**2 * time_years) / (
            mid_iv * math.sqrt(time_years)
        )
        delta = _normal_cdf(d1) if option_type == "call" else _normal_cdf(d1) - 1
        candidates.append(
            {
                "symbol": row.symbol,
                "type": option_type,
                "strike": strike,
                "expiry": expiry_utc,
                "dte": time_years * 365,
                "bid_btc": bid,
                "ask_btc": ask,
                "mid_btc": mid,
                "bid_amount": float(row.bid_amount),
                "ask_amount": float(row.ask_amount),
                "mid_iv": mid_iv,
                "forward_delta": delta,
            }
        )
    frame = pd.DataFrame(candidates)
    if frame.empty:
        return frame
    picks = []
    for option_type, target in (("call", target_delta), ("put", -target_delta)):
        side = frame.loc[frame["type"] == option_type]
        if side.empty:
            return pd.DataFrame(columns=frame.columns)
        picks.append(side.loc[[(side["forward_delta"] - target).abs().idxmin()]])
    return pd.concat(picks).sort_values("type").reset_index(drop=True)


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
