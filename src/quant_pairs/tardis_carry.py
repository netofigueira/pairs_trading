"""Hold-to-expiry straddle carry: Tardis executable entry, official settlement exit.

Entry uses real Tardis top-of-book asks on a free monthly sample day. The
position is held to expiry, so no option exit quote is needed: the payoff comes
from Deribit's official delivery price. The optional static hedge is sized once
from the observed entry delta, held to expiry, closed at the delivery price and
charged hourly funding from public history.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.funding import funding_pnl_btc
from quant_pairs.settlement import (
    delivery_price_on,
    settlement_fee_btc,
    settlement_payoff_btc,
)
from quant_pairs.tardis_intraday import (
    DEFAULT_PERP_TAKER_FEE_RATE,
    PERP_CONTRACT_SIZE_USD,
    _mid,
    _option_fee,
    _perp_book,
    _utc,
    read_option_deltas,
)
from quant_pairs.tardis_options import select_atm_straddle
from quant_pairs.tardis_quotes import reconstruct_top_of_book


def run_carry_straddle(
    option_quotes_path: Path | str,
    perp_quotes_path: Path | str,
    *,
    entry_at: pd.Timestamp,
    delivery_prices: pd.DataFrame,
    max_age: pd.Timedelta = pd.Timedelta(minutes=5),
    min_dte: int = 7,
    max_dte: int = 30,
    target_dte: float = 14.0,
    contracts: float = 1.0,
    options_chain_path: Path | str | None = None,
    funding: pd.DataFrame | None = None,
    perp_taker_fee_rate: float = DEFAULT_PERP_TAKER_FEE_RATE,
    hedge_exit_slippage_bps: float = 0.0,
) -> dict[str, object]:
    """Simulate one long ATM straddle bought at real asks and held to expiry.

    Without ``options_chain_path`` the result is the unhedged variant. With it,
    ``funding`` becomes mandatory: the static hedge's carry cannot be stated
    honestly without its funding leg.
    """

    entry = _utc(entry_at)
    if max_age < pd.Timedelta(0):
        raise ValueError("max_age cannot be negative")
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    if options_chain_path is not None and funding is None:
        raise ValueError("a static hedge held to expiry requires funding history")

    entry_options = reconstruct_top_of_book(
        option_quotes_path, as_of=entry, max_age=max_age
    )
    entry_perp = _perp_book(perp_quotes_path, as_of=entry, max_age=max_age)
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
    if entry_legs["ask_amount"].lt(contracts).any():
        raise ValueError("top-of-book option size is smaller than the requested contracts")

    expiry = pd.Timestamp(selected["expiry"].iloc[0])
    delivery_price = delivery_price_on(delivery_prices, expiry)

    legs = []
    payoff_total = 0.0
    settlement_fees = 0.0
    for symbol in symbols:
        parsed = selected.loc[selected["symbol"] == symbol].iloc[0]
        payoff = settlement_payoff_btc(
            str(parsed["type"]), float(parsed["strike"]), delivery_price
        )
        payoff_total += payoff * contracts
        settlement_fees += settlement_fee_btc(payoff) * contracts
        legs.append(
            {
                "symbol": symbol,
                "type": str(parsed["type"]),
                "strike": float(parsed["strike"]),
                "expiry": str(parsed["expiry"]),
                "entry_ask_btc": float(entry_legs.loc[symbol, "ask_price"]),
                "settlement_payoff_btc": payoff,
            }
        )

    entry_ask = float(entry_legs["ask_price"].sum()) * contracts
    entry_fees = contracts * sum(
        _option_fee(float(price)) for price in entry_legs["ask_price"].tolist()
    )
    net_unhedged = payoff_total - entry_ask - entry_fees - settlement_fees

    result: dict[str, object] = {
        "status": "carry_unhedged_settled",
        "entry_at": str(entry),
        "expiry_at": str(expiry),
        "days_held": (expiry - entry).total_seconds() / 86_400,
        "max_age_seconds": max_age.total_seconds(),
        "contracts_per_leg": contracts,
        "entry_underlying_mid_usd": underlying_mid,
        "delivery_price_usd": delivery_price,
        "legs": legs,
        "entry_premium_btc": entry_ask,
        "settlement_payoff_btc": payoff_total,
        "option_entry_fees_btc": entry_fees,
        "settlement_fees_btc": settlement_fees,
        "net_unhedged_pnl_btc": net_unhedged,
        "hedge_contracts": None,
        "hedge_pnl_btc": None,
        "hedge_fees_btc": None,
        "funding_pnl_btc": None,
        "net_static_hedged_pnl_btc": None,
    }
    if options_chain_path is None:
        return result

    deltas = read_option_deltas(
        options_chain_path, symbols=symbols, as_of=entry, max_age=max_age
    )
    option_delta_btc = sum(deltas[symbol] for symbol in symbols) * contracts
    hedge_contracts = round(-option_delta_btc * underlying_mid / PERP_CONTRACT_SIZE_USD)
    hedge = _static_hedge_accounting(
        hedge_contracts,
        entry_perp=entry_perp,
        exit_price=delivery_price,
        taker_fee_rate=perp_taker_fee_rate,
        exit_slippage_bps=hedge_exit_slippage_bps,
    )
    funding_pnl = funding_pnl_btc(
        funding, contracts=hedge_contracts, start=entry, end=expiry
    )
    result.update(
        {
            "status": "carry_static_hedged_settled",
            "entry_option_delta_btc": option_delta_btc,
            "entry_option_deltas": deltas,
            "hedge_contracts": hedge_contracts,
            "entry_residual_delta_btc": (
                option_delta_btc + hedge_contracts * PERP_CONTRACT_SIZE_USD / underlying_mid
            ),
            "hedge_exit_price_source": "delivery_price",
            "hedge_exit_slippage_bps": hedge_exit_slippage_bps,
            "hedge_pnl_btc": hedge["pnl_btc"],
            "hedge_fees_btc": hedge["fees_btc"],
            "funding_pnl_btc": funding_pnl,
            "net_static_hedged_pnl_btc": (
                net_unhedged + hedge["pnl_btc"] - hedge["fees_btc"] + funding_pnl
            ),
        }
    )
    return result


def _static_hedge_accounting(
    contracts: int,
    *,
    entry_perp: pd.Series,
    exit_price: float,
    taker_fee_rate: float,
    exit_slippage_bps: float,
) -> dict[str, float]:
    """Inverse-perp P&L for a hedge opened crossing the spread and closed at settlement.

    The exit fill is the official delivery price (index TWAP at expiry), a
    declared approximation because free samples carry no perp book at expiry.
    ``exit_slippage_bps`` shifts the exit fill against the position to stress
    the missing exit spread.
    """

    if taker_fee_rate < 0:
        raise ValueError("perp taker fee rate cannot be negative")
    if exit_slippage_bps < 0:
        raise ValueError("exit slippage cannot be negative")
    if contracts == 0:
        return {"pnl_btc": 0.0, "fees_btc": 0.0}
    side = "ask" if contracts > 0 else "bid"
    entry_fill = float(entry_perp[f"{side}_price"])
    # closing a long sells (adverse: lower price); closing a short buys (higher)
    exit_price *= 1 + (-1 if contracts > 0 else 1) * exit_slippage_bps / 10_000
    notional_usd = abs(contracts) * PERP_CONTRACT_SIZE_USD
    if notional_usd > float(entry_perp[f"{side}_amount"]):
        raise ValueError("top-of-book perp size is smaller than the delta hedge")
    pnl = contracts * PERP_CONTRACT_SIZE_USD * (1 / entry_fill - 1 / exit_price)
    fees = notional_usd * taker_fee_rate * (1 / entry_fill + 1 / exit_price)
    return {"pnl_btc": pnl, "fees_btc": fees}
