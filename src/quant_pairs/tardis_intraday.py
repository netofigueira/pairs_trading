"""Executable intraday straddle plumbing using Tardis Deribit samples."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.funding import funding_pnl_btc
from quant_pairs.tardis_options import select_atm_straddle
from quant_pairs.tardis_quotes import reconstruct_top_of_book

OPTION_FEE_BTC_PER_CONTRACT = 0.0003
OPTION_FEE_PREMIUM_CAP = 0.125
PERP_CONTRACT_SIZE_USD = 10.0
DEFAULT_PERP_TAKER_FEE_RATE = 0.0005


def run_intraday_straddle(
    option_quotes_path: Path | str,
    perp_quotes_path: Path | str,
    *,
    entry_at: pd.Timestamp,
    exit_at: pd.Timestamp,
    max_age: pd.Timedelta = pd.Timedelta(minutes=5),
    min_dte: int = 7,
    max_dte: int = 30,
    target_dte: float = 14.0,
    contracts: float = 1.0,
    options_chain_path: Path | str | None = None,
    perp_taker_fee_rate: float = DEFAULT_PERP_TAKER_FEE_RATE,
    funding: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Simulate one long ATM straddle with executable top-of-book fills.

    Without ``options_chain_path``, the result deliberately declares the missing
    delta hedge. With it, the entry delta observed by Tardis is neutralized once
    using integer BTC-PERPETUAL contracts and both hedge fills cross the spread.
    """

    entry = _utc(entry_at)
    exit_ = _utc(exit_at)
    if entry.date() != exit_.date() or exit_ <= entry:
        raise ValueError("entry_at and exit_at must be ordered within one UTC day")
    if max_age < pd.Timedelta(0):
        raise ValueError("max_age cannot be negative")
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    if perp_taker_fee_rate < 0:
        raise ValueError("perp_taker_fee_rate cannot be negative")

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
        target_dte=target_dte,
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

    result: dict[str, object] = {
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
        "funding_pnl_btc": None,
        "net_delta_hedged_before_funding_btc": None,
        "net_delta_hedged_pnl_btc": None,
    }
    if options_chain_path is not None:
        deltas = read_option_deltas(
            options_chain_path,
            symbols=symbols,
            as_of=entry,
            max_age=max_age,
        )
        option_delta_btc = sum(deltas[symbol] for symbol in symbols) * contracts
        hedge_contracts = round(-option_delta_btc * underlying_mid / PERP_CONTRACT_SIZE_USD)
        hedge = _hedge_accounting(
            hedge_contracts,
            entry_perp=entry_perp,
            exit_perp=exit_perp,
            taker_fee_rate=perp_taker_fee_rate,
        )
        total_gross = gross_mid_pnl + hedge["gross_mid_pnl_btc"]
        total_spread = spread_cost + hedge["spread_cost_btc"]
        total_fees = option_fees + hedge["fees_btc"]
        result.update(
            {
                "status": "delta_hedged_intraday_plumbing_missing_funding",
                "entry_option_delta_btc": option_delta_btc,
                "entry_option_deltas": deltas,
                "hedge_contracts": hedge_contracts,
                "entry_residual_delta_btc": (
                    option_delta_btc
                    + hedge_contracts * PERP_CONTRACT_SIZE_USD / underlying_mid
                ),
                "delta_hedge_pnl_btc": hedge["executable_pnl_before_fees_btc"],
                "delta_hedge_fees_btc": hedge["fees_btc"],
                "delta_hedge_spread_cost_btc": hedge["spread_cost_btc"],
                "total_gross_mid_pnl_btc": total_gross,
                "total_spread_cost_btc": total_spread,
                "total_fees_btc": total_fees,
                "net_delta_hedged_before_funding_btc": (
                    total_gross - total_spread - total_fees
                ),
            }
        )
        if funding is not None:
            funding_pnl = funding_pnl_btc(
                funding, contracts=hedge_contracts, start=entry, end=exit_
            )
            result.update(
                {
                    "status": "delta_hedged_intraday_with_funding",
                    "funding_pnl_btc": funding_pnl,
                    "net_delta_hedged_pnl_btc": (
                        total_gross - total_spread - total_fees + funding_pnl
                    ),
                }
            )
    return result


def read_option_deltas(
    path: Path | str,
    *,
    symbols: list[str],
    as_of: pd.Timestamp,
    max_age: pd.Timedelta,
    chunk_rows: int = 1_000_000,
) -> dict[str, float]:
    """Read fresh exchange deltas for selected contracts from a sorted chain CSV."""

    if not symbols:
        raise ValueError("symbols cannot be empty")
    if chunk_rows <= 0 or max_age < pd.Timedelta(0):
        raise ValueError("invalid chunk_rows or max_age")
    cutoff_us = int(_utc(as_of).timestamp() * 1_000_000)
    oldest_us = cutoff_us - int(max_age.total_seconds() * 1_000_000)
    wanted = set(symbols)
    matches: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=["symbol", "timestamp", "local_timestamp", "delta"],
        chunksize=chunk_rows,
    ):
        timestamps = pd.to_numeric(chunk["timestamp"], errors="coerce")
        local_timestamps = pd.to_numeric(chunk["local_timestamp"], errors="coerce")
        if local_timestamps.min() > cutoff_us:
            break
        selected = chunk.loc[
            chunk["symbol"].isin(wanted)
            & timestamps.between(oldest_us, cutoff_us)
            & local_timestamps.le(cutoff_us)
            & chunk["delta"].notna(),
            ["symbol", "timestamp", "delta"],
        ]
        if not selected.empty:
            matches.append(selected)
    if not matches:
        raise ValueError(f"no fresh option deltas at {as_of}")
    latest = (
        pd.concat(matches, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("symbol", keep="last")
        .set_index("symbol")["delta"]
    )
    missing = wanted.difference(latest.index)
    if missing:
        raise ValueError(f"missing fresh deltas for selected contracts: {sorted(missing)}")
    return {symbol: float(latest.loc[symbol]) for symbol in symbols}


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


def _hedge_accounting(
    contracts: int,
    *,
    entry_perp: pd.Series,
    exit_perp: pd.Series,
    taker_fee_rate: float,
) -> dict[str, float]:
    if contracts >= 0:
        entry_fill = float(entry_perp["ask_price"])
        exit_fill = float(exit_perp["bid_price"])
        available_entry = float(entry_perp["ask_amount"])
        available_exit = float(exit_perp["bid_amount"])
    else:
        entry_fill = float(entry_perp["bid_price"])
        exit_fill = float(exit_perp["ask_price"])
        available_entry = float(entry_perp["bid_amount"])
        available_exit = float(exit_perp["ask_amount"])
    notional_usd = abs(contracts) * PERP_CONTRACT_SIZE_USD
    if notional_usd > min(available_entry, available_exit):
        raise ValueError("top-of-book perp size is smaller than the delta hedge")
    executable_pnl = contracts * PERP_CONTRACT_SIZE_USD * (1 / entry_fill - 1 / exit_fill)
    entry_mid = _mid(entry_perp)
    exit_mid = _mid(exit_perp)
    gross_mid_pnl = contracts * PERP_CONTRACT_SIZE_USD * (1 / entry_mid - 1 / exit_mid)
    fees = notional_usd * taker_fee_rate * (1 / entry_fill + 1 / exit_fill)
    return {
        "gross_mid_pnl_btc": gross_mid_pnl,
        "spread_cost_btc": gross_mid_pnl - executable_pnl,
        "executable_pnl_before_fees_btc": executable_pnl,
        "fees_btc": fees,
    }


def _option_fee(fill_price_btc: float) -> float:
    return min(OPTION_FEE_BTC_PER_CONTRACT, OPTION_FEE_PREMIUM_CAP * fill_price_btc)


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")
