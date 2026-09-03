"""Daily ATM straddle selection from the public Deribit option trade tape.

Prints are real executions, not our fills: the central price per leg is the
trade closest to the decision time, and short-entry scenarios subtract the
half-spread calibrated on the quarterly Tardis books.  Days without a pairable
call/put print at one strike and tenor are coverage gaps, never interpolation.
"""

from __future__ import annotations

import pandas as pd

from quant_pairs.inverse_options import implied_volatility
from quant_pairs.tardis_options import _parse_option


def select_daily_straddle_prints(
    trades: pd.DataFrame,
    *,
    decision_at: pd.Timestamp,
    min_dte: float = 7.0,
    max_dte: float = 30.0,
    target_dte: float = 14.0,
    max_age: pd.Timedelta = pd.Timedelta(hours=2),
) -> pd.DataFrame:
    """Pick one call and one put print at a common ATM strike near the target DTE.

    Strictly causal: only prints with ``traded_at <= decision_at`` and no older
    than ``max_age`` are eligible; per leg, the LAST pre-decision print wins.
    ``trades`` needs columns ``instrument_name, traded_at, price, iv,
    index_price``.  Tenor is chosen first (expiry nearest ``target_dte`` among
    those with both sides pairable), then the strike nearest the index price of
    the latest pre-decision print.  Returns an empty frame when no pair exists.
    """

    required = {"instrument_name", "traded_at", "price", "iv", "index_price"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"trades are missing required columns: {sorted(missing)}")
    decision = _utc(decision_at)
    if not min_dte <= target_dte <= max_dte:
        raise ValueError("target_dte must lie within the DTE bounds")
    if max_age <= pd.Timedelta(0):
        raise ValueError("max_age must be positive")

    frame = trades.copy()
    frame["traded_at"] = pd.to_datetime(frame["traded_at"], utc=True)
    frame = frame.loc[
        (frame["traded_at"] <= decision) & (frame["traded_at"] >= decision - max_age)
    ]
    if frame.empty:
        return pd.DataFrame()
    parsed = frame["instrument_name"].map(_parse_option)
    frame = frame.loc[parsed.notna()].copy()
    frame[["expiry", "type", "strike"]] = pd.DataFrame(parsed.dropna().tolist(), index=frame.index)
    frame["expiry"] = pd.to_datetime(frame["expiry"], utc=True)
    frame["dte"] = (frame["expiry"] - decision).dt.total_seconds() / 86_400
    frame = frame.loc[(frame["dte"] >= min_dte) & (frame["dte"] <= max_dte) & frame["price"].gt(0)]
    if frame.empty:
        return pd.DataFrame()

    # ATM reference: the index price of the latest pre-decision print.
    reference = frame.loc[frame["traded_at"].idxmax(), "index_price"]
    pairable = frame.groupby(["expiry", "strike"])["type"].nunique()
    pairable = pairable[pairable == 2].reset_index()
    if pairable.empty:
        return pd.DataFrame()
    pairable["dte"] = (
        pd.to_datetime(pairable["expiry"], utc=True) - decision
    ).dt.total_seconds() / 86_400
    chosen_expiry = pairable.loc[(pairable["dte"] - target_dte).abs().idxmin(), "expiry"]
    at_expiry = pairable.loc[pairable["expiry"] == chosen_expiry]
    chosen_strike = at_expiry.loc[
        (at_expiry["strike"] / float(reference) - 1).abs().idxmin(), "strike"
    ]

    legs = []
    for option_type in ("call", "put"):
        side = frame.loc[
            (frame["expiry"] == chosen_expiry)
            & (frame["strike"] == chosen_strike)
            & (frame["type"] == option_type)
        ]
        pick = side.loc[side["traded_at"].idxmax()]
        legs.append(
            {
                "instrument_name": pick["instrument_name"],
                "type": option_type,
                "strike": float(chosen_strike),
                "expiry": pd.Timestamp(chosen_expiry),
                "dte": float((pd.Timestamp(chosen_expiry) - decision).total_seconds() / 86_400),
                "traded_at": pick["traded_at"],
                "seconds_from_decision": float((decision - pick["traded_at"]).total_seconds()),
                "print_price_btc": float(pick["price"]),
                "print_iv": float(pick["iv"]) / 100.0,
                "index_price": float(pick["index_price"]),
            }
        )
    return pd.DataFrame(legs)


def short_entry_from_prints(
    legs: pd.DataFrame,
    *,
    relative_half_spread: float,
    contracts: float,
) -> dict[str, object]:
    """Short-entry economics: sell each leg at print price minus the half-spread.

    Bid IVs are re-inverted from the discounted prices against each leg's own
    index price as the forward proxy (declared: basis is not observable in the
    tape).  Returns the credit, per-leg bid IVs and the mean bid variance the
    frozen gate compares against.
    """

    if relative_half_spread < 0 or contracts <= 0:
        raise ValueError("half-spread must be non-negative and contracts positive")
    if len(legs) != 2:
        raise ValueError("legs must hold exactly one call and one put")
    rows = []
    credit = 0.0
    for leg in legs.itertuples(index=False):
        bid_price = float(leg.print_price_btc) * (1.0 - relative_half_spread)
        bid_iv = implied_volatility(
            str(leg.type),
            price_btc=bid_price,
            forward=float(leg.index_price),
            strike=float(leg.strike),
            time_years=float(leg.dte) / 365.0,
        )
        credit += bid_price * contracts
        rows.append(
            {
                "type": str(leg.type),
                "strike": float(leg.strike),
                "entry_iv": float(leg.print_iv),
                "bid_price_btc": bid_price,
                "bid_iv": bid_iv,
            }
        )
    return {
        "legs": rows,
        "entry_credit_btc": credit,
        "mean_bid_variance": sum(row["bid_iv"] ** 2 for row in rows) / 2.0,
    }


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("decision_at must be timezone-aware")
    return timestamp.tz_convert("UTC")
