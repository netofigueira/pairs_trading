"""Executable intraday straddle plumbing using Tardis Deribit samples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.tardis_options import select_atm_straddle
from quant_pairs.tardis_quotes import reconstruct_top_of_book

OPTION_FEE_BTC_PER_CONTRACT = 0.0003
OPTION_FEE_PREMIUM_CAP = 0.125


def run_intraday_straddle(
    option_quotes_path: Path | str,
    perp_quotes_path: Path | str,
    *,
    entry_at: pd.Timestamp,
    exit_at: pd.Timestamp,
    max_age: pd.Timedelta = pd.Timedelta(minutes=5),
    min_dte: int = 7,
    max_dte: int = 30,
    contracts: float = 1.0,
) -> dict[str, object]:
    """Simulate one unhedged long ATM straddle with executable top-of-book fills.

    The result deliberately declares the missing delta hedge. It validates quote
    synchronization, contract continuity, spread and option fees; it must not be
    interpreted as a strategy backtest until exchange Greeks are joined.
    """

    entry = _utc(entry_at)
    exit_ = _utc(exit_at)
    if entry.date() != exit_.date() or exit_ <= entry:
        raise ValueError("entry_at and exit_at must be ordered within one UTC day")
    if max_age < pd.Timedelta(0):
        raise ValueError("max_age cannot be negative")
    if contracts <= 0:
        raise ValueError("contracts must be positive")

    entry_options = reconstruct_top_of_book(
        option_quotes_path, as_of=entry, max_age=max_age
    )
    exit_options = reconstruct_top_of_book(option_quotes_path, as_of=exit_, max_age=max_age)
    entry_perp = _perp_book(perp_quotes_path, as_of=entry, max_age=max_age)
    exit_perp = _perp_book(perp_quotes_path, as_of=exit_, max_age=max_age)
    underlying_mid = _mid(entry_perp)

    btc_options = entry_options.loc[entry_options["symbol"].str.startswith("BTC-")]
    selected = select_atm_straddle(
        btc_options,
        underlying_mid=underlying_mid,
        as_of=entry,
        min_dte=min_dte,
        max_dte=max_dte,
    )
    if len(selected) != 2:
        raise ValueError("no executable BTC ATM call/put pair in the requested DTE range")
    symbols = selected["symbol"].tolist()
    entry_legs = entry_options.set_index("symbol").reindex(symbols)
    exit_legs = exit_options.set_index("symbol").reindex(symbols)
    if exit_legs[["bid_price", "bid_amount"]].isna().any().any():
        raise ValueError("the selected entry contracts lack fresh executable exit bids")
    if entry_legs["ask_amount"].lt(contracts).any() or exit_legs["bid_amount"].lt(contracts).any():
        raise ValueError("top-of-book option size is smaller than the requested contracts")

    entry_ask = float(entry_legs["ask_price"].sum()) * contracts
    exit_bid = float(exit_legs["bid_price"].sum()) * contracts
    entry_mid = float(((entry_legs["ask_price"] + entry_legs["bid_price"]) / 2).sum())
    exit_mid = float(((exit_legs["ask_price"] + exit_legs["bid_price"]) / 2).sum())
    gross_mid_pnl = (exit_mid - entry_mid) * contracts
    executable_pnl = exit_bid - entry_ask
    spread_cost = gross_mid_pnl - executable_pnl
    option_fees = contracts * sum(
        _option_fee(float(price))
        for price in (*entry_legs["ask_price"].tolist(), *exit_legs["bid_price"].tolist())
    )

    legs = []
    for symbol in symbols:
        parsed = selected.loc[selected["symbol"] == symbol].iloc[0]
        legs.append(
            {
                "symbol": symbol,
                "type": str(parsed["type"]),
                "strike": float(parsed["strike"]),
                "expiry": str(parsed["expiry"]),
                "entry_ask_btc": float(entry_legs.loc[symbol, "ask_price"]),
                "exit_bid_btc": float(exit_legs.loc[symbol, "bid_price"]),
            }
        )

    return {
        "status": "plumbing_only_missing_delta_hedge",
        "entry_at": str(entry),
        "exit_at": str(exit_),
        "max_age_seconds": max_age.total_seconds(),
        "contracts_per_leg": contracts,
        "entry_underlying_mid_usd": underlying_mid,
        "exit_underlying_mid_usd": _mid(exit_perp),
        "legs": legs,
        "gross_mid_pnl_btc": gross_mid_pnl,
        "spread_cost_btc": spread_cost,
        "executable_pnl_before_fees_btc": executable_pnl,
        "option_fees_btc": option_fees,
        "net_unhedged_pnl_btc": executable_pnl - option_fees,
        "delta_hedge_pnl_btc": None,
        "net_delta_hedged_pnl_btc": None,
    }


def _perp_book(
    path: Path | str, *, as_of: pd.Timestamp, max_age: pd.Timedelta
) -> pd.Series:
    books = reconstruct_top_of_book(path, as_of=as_of, max_age=max_age)
    selected = books.loc[books["symbol"] == "BTC-PERPETUAL"]
    if len(selected) != 1:
        raise ValueError(f"no fresh executable BTC-PERPETUAL book at {as_of}")
    return selected.iloc[0]


def _mid(book: pd.Series) -> float:
    return (float(book["bid_price"]) + float(book["ask_price"])) / 2


def _option_fee(fill_price_btc: float) -> float:
    return min(OPTION_FEE_BTC_PER_CONTRACT, OPTION_FEE_PREMIUM_CAP * fill_price_btc)


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")
