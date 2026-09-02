"""Daily delta-hedged short option baskets over synthetic inverse-option marks.

The option legs are held to expiry and settle at the official delivery price,
exactly like the unhedged quarterly envelope.  The hedge leg is an inverse perp
position rebalanced at each daily synthetic mark to neutralize the basket's
BTC-value delta, paying taker fees on every rebalance and real hourly funding
on the held notional.  All marks are synthetic_model provenance: this is an
envelope, not a reconstruction of observed fills.

Legs are ``{"type", "strike", "entry_iv"}`` mappings: two legs at one strike
form the straddle of the Phase 1 study; different strikes form a strangle.
Each leg's IV path follows the change in DVOL anchored to its own entry IV,
the same convention as ``build_daily_straddle_marks``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd

from quant_pairs.funding import PERP_CONTRACT_SIZE_USD, funding_pnl_btc
from quant_pairs.inverse_options import inverse_option_price
from quant_pairs.settlement import settlement_fee_btc, settlement_payoff_btc
from quant_pairs.synthetic_option_backfill import (
    _available_daily_closes,
    _last_available,
)
from quant_pairs.tardis_intraday import DEFAULT_PERP_TAKER_FEE_RATE

_RELATIVE_BUMP = 1e-4
_MIN_IV = 0.01

Leg = Mapping[str, object]


def basket_value_btc(
    legs: Sequence[Leg],
    *,
    forward: float,
    time_years: float,
    iv_shift: float = 0.0,
) -> float:
    """BTC value of one contract of each leg, IVs shifted additively."""

    return sum(
        inverse_option_price(
            str(leg["type"]),
            forward=forward,
            strike=float(leg["strike"]),
            time_years=time_years,
            volatility=max(float(leg["entry_iv"]) + iv_shift, _MIN_IV),
        )
        for leg in legs
    )


def basket_delta_btc(
    legs: Sequence[Leg],
    *,
    underlying: float,
    forward: float,
    time_years: float,
    iv_shift: float = 0.0,
) -> float:
    """d(basket BTC value)/d(underlying USD), forward scaling with the spot.

    The synthetic marks carry the forward as ``underlying * exp(basis * tau)``,
    so a spot bump moves the forward proportionally.  Central finite difference
    over the exact pricer keeps the hedge consistent with the marking model.
    """

    if underlying <= 0 or forward <= 0:
        raise ValueError("underlying and forward must be positive")
    if time_years <= 0:
        raise ValueError("time_years must be positive")
    bump = underlying * _RELATIVE_BUMP
    shifted_values = [
        basket_value_btc(
            legs,
            forward=forward * shifted / underlying,
            time_years=time_years,
            iv_shift=iv_shift,
        )
        for shifted in (underlying + bump, underlying - bump)
    ]
    return (shifted_values[0] - shifted_values[1]) / (2 * bump)


def straddle_delta_btc(
    *,
    underlying: float,
    forward: float,
    strike: float,
    time_years: float,
    volatility: float,
) -> float:
    """Delta of the one-strike call/put pair; kept as the Phase 1 entry point."""

    if strike <= 0 or volatility <= 0:
        raise ValueError("strike and volatility must be positive")
    legs = [
        {"type": option_type, "strike": strike, "entry_iv": volatility}
        for option_type in ("call", "put")
    ]
    return basket_delta_btc(legs, underlying=underlying, forward=forward, time_years=time_years)


def simulate_delta_hedged_short_basket(
    prices: pd.DataFrame,
    dvol: pd.DataFrame,
    *,
    entry_at: pd.Timestamp,
    expiry_at: pd.Timestamp,
    legs: Sequence[Leg],
    contracts: float,
    entry_underlying: float,
    entry_forward: float,
    entry_credit_btc: float,
    entry_fees_btc: float,
    delivery_price: float,
    funding: pd.DataFrame,
    perp_taker_fee_rate: float = DEFAULT_PERP_TAKER_FEE_RATE,
    perp_contract_size_usd: float = PERP_CONTRACT_SIZE_USD,
    annualization_days: int = 365,
) -> dict[str, object]:
    """Hold a short option basket to expiry, delta-hedging daily with the perp.

    Daily marks are built internally from the price and DVOL daily bars using
    the same availability, basis-carry and DVOL-anchored IV conventions as
    ``build_daily_straddle_marks``.  Funding must cover the (entry, expiry]
    window at hourly resolution.
    """

    entry = _utc(entry_at)
    expiry = _utc(expiry_at)
    if expiry <= entry:
        raise ValueError("expiry_at must be after entry_at")
    if not legs:
        raise ValueError("at least one leg is required")
    for leg in legs:
        if float(leg["strike"]) <= 0 or float(leg["entry_iv"]) <= 0:
            raise ValueError("leg strikes and entry IVs must be positive")
    if contracts <= 0 or delivery_price <= 0:
        raise ValueError("contracts and delivery_price must be positive")
    if entry_underlying <= 0 or entry_forward <= 0:
        raise ValueError("entry underlying and forward must be positive")
    if entry_credit_btc <= 0 or entry_fees_btc < 0:
        raise ValueError("entry credit must be positive and fees non-negative")
    if entry != entry.floor("h") or expiry != expiry.floor("h"):
        raise ValueError("entry and expiry must be aligned to full UTC hours")

    total_years = (expiry - entry).total_seconds() / (annualization_days * 86_400)
    basis_yield = math.log(entry_forward / entry_underlying) / total_years

    points = [
        {
            "at": entry,
            "underlying": float(entry_underlying),
            "delta": basket_delta_btc(
                legs,
                underlying=entry_underlying,
                forward=entry_forward,
                time_years=total_years,
            ),
            "basket_mid_btc": basket_value_btc(legs, forward=entry_forward, time_years=total_years),
        }
    ]
    price_panel = _available_daily_closes(prices, value_name="underlying")
    dvol_panel = _available_daily_closes(dvol, value_name="dvol_points")
    entry_dvol = _last_available(dvol_panel, entry, "dvol_points") / 100
    decisions = price_panel.loc[
        (price_panel["available_at"] > entry) & (price_panel["available_at"] < expiry)
    ].copy()
    if not decisions.empty:
        decisions = pd.merge_asof(
            decisions.sort_values("available_at"),
            dvol_panel.sort_values("available_at"),
            on="available_at",
            direction="backward",
        ).dropna(subset=["dvol_points"])
    for row in decisions.itertuples(index=False):
        remaining_years = (expiry - row.available_at).total_seconds() / (
            annualization_days * 86_400
        )
        forward = float(row.underlying) * math.exp(basis_yield * remaining_years)
        iv_shift = float(row.dvol_points) / 100 - entry_dvol
        points.append(
            {
                "at": _utc(pd.Timestamp(row.available_at)),
                "underlying": float(row.underlying),
                "delta": basket_delta_btc(
                    legs,
                    underlying=float(row.underlying),
                    forward=forward,
                    time_years=remaining_years,
                    iv_shift=iv_shift,
                ),
                "basket_mid_btc": basket_value_btc(
                    legs, forward=forward, time_years=remaining_years, iv_shift=iv_shift
                ),
            }
        )

    return _run_hedged_book(
        points,
        legs=legs,
        expiry=expiry,
        contracts=contracts,
        entry_credit_btc=entry_credit_btc,
        entry_fees_btc=entry_fees_btc,
        delivery_price=delivery_price,
        funding=funding,
        perp_taker_fee_rate=perp_taker_fee_rate,
        perp_contract_size_usd=perp_contract_size_usd,
    )


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
    """Phase 1 entry point: one-strike short straddle over pre-built marks.

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

    legs = [
        {"type": option_type, "strike": strike, "entry_iv": entry_iv}
        for option_type in ("call", "put")
    ]
    total_years = (expiry - entry).total_seconds() / (365 * 86_400)
    points = [
        {
            "at": entry,
            "underlying": float(entry_underlying),
            "delta": basket_delta_btc(
                legs,
                underlying=entry_underlying,
                forward=entry_forward,
                time_years=total_years,
            ),
            "basket_mid_btc": basket_value_btc(legs, forward=entry_forward, time_years=total_years),
        }
    ]
    if not marks.empty:
        ordered = marks.sort_values("decision_at")
        for row in ordered.itertuples(index=False):
            remaining_years = float(row.remaining_dte) / 365
            iv_shift = float(row.modeled_iv) - entry_iv
            points.append(
                {
                    "at": _utc(pd.Timestamp(row.decision_at)),
                    "underlying": float(row.underlying_usd),
                    "delta": basket_delta_btc(
                        legs,
                        underlying=float(row.underlying_usd),
                        forward=float(row.forward_usd),
                        time_years=remaining_years,
                        iv_shift=iv_shift,
                    ),
                    "basket_mid_btc": float(row.close_mid_btc),
                }
            )
    return _run_hedged_book(
        points,
        legs=legs,
        expiry=expiry,
        contracts=contracts,
        entry_credit_btc=entry_credit_btc,
        entry_fees_btc=entry_fees_btc,
        delivery_price=delivery_price,
        funding=funding,
        perp_taker_fee_rate=perp_taker_fee_rate,
        perp_contract_size_usd=perp_contract_size_usd,
    )


def _run_hedged_book(
    points: list[dict[str, object]],
    *,
    legs: Sequence[Leg],
    expiry: pd.Timestamp,
    contracts: float,
    entry_credit_btc: float,
    entry_fees_btc: float,
    delivery_price: float,
    funding: pd.DataFrame,
    perp_taker_fee_rate: float,
    perp_contract_size_usd: float,
) -> dict[str, object]:
    hedge_trading_pnl = 0.0
    hedge_fees = 0.0
    funding_pnl = 0.0
    previous_notional = 0.0
    daily_rows: list[dict[str, object]] = []
    for index, point in enumerate(points):
        # A short basket carries -contracts * delta of BTC-value exposure;
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
                "short_straddle_mid_btc": -float(point["basket_mid_btc"]) * contracts,
            }
        )
        previous_notional = notional

    # Close the hedge at the delivery price.
    hedge_fees += perp_taker_fee_rate * abs(previous_notional) / delivery_price

    payoff_per_contract = sum(
        settlement_payoff_btc(str(leg["type"]), float(leg["strike"]), delivery_price)
        for leg in legs
    )
    settlement_fees = (
        sum(
            settlement_fee_btc(
                settlement_payoff_btc(str(leg["type"]), float(leg["strike"]), delivery_price)
            )
            for leg in legs
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
