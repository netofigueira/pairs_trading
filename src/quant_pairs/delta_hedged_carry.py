"""Daily delta-hedged short straddle carry over synthetic inverse-option marks.

The option leg is held to expiry and settles at the official delivery price,
exactly like the unhedged quarterly envelope.  The hedge leg is an inverse perp
position rebalanced at each daily synthetic mark to neutralize the straddle's
BTC-value delta, paying taker fees on every rebalance and real hourly funding
on the held notional.  All marks are synthetic_model provenance: this is an
envelope, not a reconstruction of observed fills.
"""

from __future__ import annotations

import pandas as pd

from quant_pairs.funding import PERP_CONTRACT_SIZE_USD, funding_pnl_btc
from quant_pairs.inverse_options import inverse_option_price
from quant_pairs.settlement import settlement_fee_btc, settlement_payoff_btc
from quant_pairs.tardis_intraday import DEFAULT_PERP_TAKER_FEE_RATE

_RELATIVE_BUMP = 1e-4


def straddle_delta_btc(
    *,
    underlying: float,
    forward: float,
    strike: float,
    time_years: float,
    volatility: float,
) -> float:
    """d(straddle BTC value)/d(underlying USD), forward scaling with the spot.

    The synthetic marks carry the forward as ``underlying * exp(basis * tau)``,
    so a spot bump moves the forward proportionally.  Central finite difference
    over the exact pricer keeps the hedge consistent with the marking model.
    """

    if underlying <= 0 or forward <= 0 or strike <= 0 or volatility <= 0:
        raise ValueError("underlying, forward, strike and volatility must be positive")
    if time_years <= 0:
        raise ValueError("time_years must be positive")
    bump = underlying * _RELATIVE_BUMP
    values = []
    for shifted in (underlying + bump, underlying - bump):
        shifted_forward = forward * shifted / underlying
        value = sum(
            inverse_option_price(
                option_type,
                forward=shifted_forward,
                strike=strike,
                time_years=time_years,
                volatility=volatility,
            )
            for option_type in ("call", "put")
        )
        values.append(value)
    return (values[0] - values[1]) / (2 * bump)


def simulate_delta_hedged_short(
    marks: pd.DataFrame,
    *,
    entry_at: pd.Timestamp,
    expiry_at: pd.Timestamp,
    strike: float,
    contracts: float,
    entry_underlying: float,
    entry_forward: float,
    entry_iv: float,
    entry_credit_btc: float,
    entry_fees_btc: float,
    delivery_price: float,
    funding: pd.DataFrame,
    perp_taker_fee_rate: float = DEFAULT_PERP_TAKER_FEE_RATE,
    perp_contract_size_usd: float = PERP_CONTRACT_SIZE_USD,
) -> dict[str, object]:
    """Hold a short straddle to expiry while delta-hedging daily with the perp.

    ``marks`` must come from ``build_daily_straddle_marks`` for the same
    contract (mid fields are used; spreads only matter to the option leg at
    entry, already embedded in ``entry_credit_btc``).  Funding must cover the
    full (entry, expiry] window at hourly resolution.
    """

    entry = _utc(entry_at)
    expiry = _utc(expiry_at)
    if expiry <= entry:
        raise ValueError("expiry_at must be after entry_at")
    if contracts <= 0 or strike <= 0 or delivery_price <= 0:
        raise ValueError("contracts, strike and delivery_price must be positive")
    if entry_credit_btc <= 0 or entry_fees_btc < 0:
        raise ValueError("entry credit must be positive and fees non-negative")
    if entry != entry.floor("h") or expiry != expiry.floor("h"):
        raise ValueError("entry and expiry must be aligned to full UTC hours")

    total_years = (expiry - entry).total_seconds() / (365 * 86_400)
    entry_delta = straddle_delta_btc(
        underlying=entry_underlying,
        forward=entry_forward,
        strike=strike,
        time_years=total_years,
        volatility=entry_iv,
    )
    entry_mid_btc = sum(
        inverse_option_price(
            option_type,
            forward=entry_forward,
            strike=strike,
            time_years=total_years,
            volatility=entry_iv,
        )
        for option_type in ("call", "put")
    )

    points = [
        {
            "at": entry,
            "underlying": float(entry_underlying),
            "delta": entry_delta,
            "straddle_mid_btc": entry_mid_btc,
        }
    ]
    if not marks.empty:
        ordered = marks.sort_values("decision_at")
        for row in ordered.itertuples(index=False):
            remaining_years = float(row.remaining_dte) / 365
            points.append(
                {
                    "at": _utc(pd.Timestamp(row.decision_at)),
                    "underlying": float(row.underlying_usd),
                    "delta": straddle_delta_btc(
                        underlying=float(row.underlying_usd),
                        forward=float(row.forward_usd),
                        strike=strike,
                        time_years=remaining_years,
                        volatility=float(row.modeled_iv),
                    ),
                    "straddle_mid_btc": float(row.close_mid_btc),
                }
            )

    hedge_trading_pnl = 0.0
    hedge_fees = 0.0
    funding_pnl = 0.0
    previous_notional = 0.0
    daily_rows: list[dict[str, object]] = []
    for index, point in enumerate(points):
        # A short straddle carries -contracts * delta of BTC-value exposure;
        # a long inverse perp of H USD contributes +H/S^2, so H = c * delta * S^2.
        raw_notional = contracts * point["delta"] * point["underlying"] ** 2
        notional = round(raw_notional / perp_contract_size_usd) * perp_contract_size_usd
        traded = notional - previous_notional
        fee = perp_taker_fee_rate * abs(traded) / point["underlying"]
        hedge_fees += fee

        next_at = points[index + 1]["at"] if index + 1 < len(points) else expiry
        next_underlying = (
            points[index + 1]["underlying"] if index + 1 < len(points) else delivery_price
        )
        segment_pnl = notional * (1 / point["underlying"] - 1 / next_underlying)
        segment_funding = 0.0
        if notional != 0.0:
            segment_funding = funding_pnl_btc(
                funding,
                contracts=notional / perp_contract_size_usd,
                start=point["at"],
                end=next_at,
                contract_size_usd=perp_contract_size_usd,
            )
        hedge_trading_pnl += segment_pnl
        funding_pnl += segment_funding
        daily_rows.append(
            {
                "at": str(point["at"]),
                "underlying_usd": point["underlying"],
                "straddle_delta": point["delta"],
                "hedge_notional_usd": notional,
                "hedge_traded_usd": traded,
                "hedge_fee_btc": fee,
                "segment_hedge_pnl_btc": segment_pnl,
                "segment_funding_btc": segment_funding,
                "short_straddle_mid_btc": -point["straddle_mid_btc"] * contracts,
            }
        )
        previous_notional = notional

    # Close the hedge at the delivery price.
    hedge_fees += perp_taker_fee_rate * abs(previous_notional) / delivery_price

    payoff_per_contract = sum(
        settlement_payoff_btc(option_type, strike, delivery_price)
        for option_type in ("call", "put")
    )
    settlement_fees = (
        sum(
            settlement_fee_btc(settlement_payoff_btc(option_type, strike, delivery_price))
            for option_type in ("call", "put")
        )
        * contracts
    )
    option_pnl = (
        entry_credit_btc - entry_fees_btc - payoff_per_contract * contracts - settlement_fees
    )
    total = option_pnl + hedge_trading_pnl + funding_pnl - hedge_fees
    return {
        "unhedged_pnl_btc": option_pnl,
        "hedged_pnl_btc": total,
        "option_pnl_btc": option_pnl,
        "hedge_trading_pnl_btc": hedge_trading_pnl,
        "hedge_fees_btc": hedge_fees,
        "funding_pnl_btc": funding_pnl,
        "rebalances": len(points),
        "daily": daily_rows,
    }


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")
