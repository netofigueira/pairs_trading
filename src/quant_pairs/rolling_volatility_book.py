"""Rolling, delta-hedged synthetic short-volatility research book.

This module deliberately separates *portfolio mechanics* from executable
historical evidence.  It can create one candidate per available forecast day,
but, until dense historical option books are available, entry IV is a declared
DVOL proxy and every mark is ``synthetic_model``.  It is useful for testing
overlap, capacity, hedge netting and margin accounting; it is not a fill
backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_pairs.delta_hedged_carry import basket_delta_btc, basket_value_btc
from quant_pairs.inverse_options import inverse_option_price
from quant_pairs.settlement import settlement_fee_btc, settlement_payoff_btc
from quant_pairs.synthetic_option_backfill import _available_daily_closes
from quant_pairs.tardis_intraday import _option_fee

_OPTION_MM_ADDON_BTC = 0.075
_PERP_MM_RATE = 0.005


@dataclass(frozen=True)
class RollingBookParameters:
    """Frozen assumptions for one synthetic rolling-book experiment."""

    horizon_days: int = 14
    contracts_per_entry: float = 0.1
    max_contracts_per_btc: float = 0.5
    bid_iv_discount_points: float = 1.0
    funding_rate_hourly: float = 0.0
    perp_taker_fee_rate: float = 0.0005
    initial_equity_btc: float = 1.0

    def __post_init__(self) -> None:
        if self.horizon_days <= 0 or self.contracts_per_entry <= 0:
            raise ValueError("horizon_days and contracts_per_entry must be positive")
        if self.max_contracts_per_btc <= 0 or self.initial_equity_btc <= 0:
            raise ValueError("book capacity and initial equity must be positive")
        if self.bid_iv_discount_points < 0 or self.perp_taker_fee_rate < 0:
            raise ValueError("discount and fee must be non-negative")


@dataclass
class _Position:
    entry_at: pd.Timestamp
    expiry_at: pd.Timestamp
    strike: float
    entry_iv: float
    entry_dvol: float
    contracts: float


def run_synthetic_rolling_short_book(
    prices: pd.DataFrame,
    dvol: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    parameters: RollingBookParameters = RollingBookParameters(),
    start_at: pd.Timestamp | None = None,
    end_at: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Run a daily-entry, 14-day rolling book over historical daily bars.

    A candidate is created only when the causal corrected GARCH forecast is
    below a *synthetic* bid-IV proxy (DVOL minus a declared IV discount).  The
    option is an ATM inverse straddle, valued daily with spot and DVOL.  The
    aggregate hedge is netted across all open positions before charging perp
    fees and funding.  An entry is skipped when it would exceed the whole-book
    contracts-per-equity cap.
    """

    panel = _market_panel(prices, dvol, forecasts, start_at=start_at, end_at=end_at)
    if panel.empty:
        raise ValueError("no aligned market/forecast days")

    last_at = panel["at"].iloc[-1]
    max_entry_at = last_at - pd.Timedelta(days=parameters.horizon_days)
    decision_panel = panel.loc[panel["at"] <= max_entry_at].copy()
    if decision_panel.empty:
        raise ValueError("no fully settled entries in selected period")

    active: list[_Position] = []
    entries: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    cash = 0.0
    hedge_notional = 0.0
    previous_underlying: float | None = None
    margin_breaches = 0

    for row in panel.itertuples(index=False):
        at = pd.Timestamp(row.at)
        underlying = float(row.underlying)
        dvol_level = float(row.dvol)
        if previous_underlying is not None and hedge_notional:
            cash += hedge_notional * (1 / previous_underlying - 1 / underlying)
            cash -= hedge_notional / previous_underlying * parameters.funding_rate_hourly * 24

        closing = [position for position in active if position.expiry_at <= at]
        for position in closing:
            payoff = _straddle_payoff(position.strike, underlying) * position.contracts
            fees = _straddle_settlement_fee(position.strike, underlying) * position.contracts
            cash -= payoff + fees
            active.remove(position)

        forecast_rv = float(row.garch_corrected_rv)
        bid_iv = max(dvol_level - parameters.bid_iv_discount_points / 100, 0.01)
        signal = at <= max_entry_at and forecast_rv**2 < bid_iv**2
        equity_before_entry = _equity(
            cash, active, underlying, dvol_level, at, parameters.initial_equity_btc
        )
        cap = parameters.max_contracts_per_btc * max(equity_before_entry, 0.0)
        open_contracts = sum(position.contracts for position in active)
        accepted = signal and open_contracts + parameters.contracts_per_entry <= cap + 1e-12
        if accepted:
            entry_iv = dvol_level
            credit = _straddle_value(underlying, underlying, entry_iv, parameters.horizon_days)
            bid_credit = _straddle_value(
                underlying, underlying, bid_iv, parameters.horizon_days
            )
            entry_fees = _straddle_entry_fee(bid_credit)
            cash += (bid_credit - entry_fees) * parameters.contracts_per_entry
            position = _Position(
                entry_at=at,
                expiry_at=at + pd.Timedelta(days=parameters.horizon_days),
                strike=underlying,
                entry_iv=entry_iv,
                entry_dvol=dvol_level,
                contracts=parameters.contracts_per_entry,
            )
            active.append(position)
            entries.append(
                {
                    "entry_at": str(at),
                    "expiry_at": str(position.expiry_at),
                    "forecast_rv": forecast_rv,
                    "dvol_mid_iv": entry_iv,
                    "synthetic_bid_iv": bid_iv,
                    "strike_usd": underlying,
                    "bid_credit_btc_per_contract": bid_credit,
                    "mid_value_btc_per_contract": credit,
                    "entry_fee_btc_per_contract": entry_fees,
                    "contracts": position.contracts,
                }
            )

        liability, total_delta = _liability_and_delta(active, underlying, dvol_level, at)
        desired_hedge = total_delta * underlying**2
        cash -= parameters.perp_taker_fee_rate * abs(desired_hedge - hedge_notional) / underlying
        hedge_notional = desired_hedge
        maintenance = (
            sum(
                position.contracts
                * (_OPTION_MM_ADDON_BTC * 2 + _position_value(position, underlying, dvol_level, at))
                for position in active
            )
            + _PERP_MM_RATE * abs(hedge_notional) / underlying
        )
        equity = parameters.initial_equity_btc + cash - liability
        breached = bool(active and equity < maintenance)
        margin_breaches += int(breached)
        daily.append(
            {
                "at": str(at),
                "source": "synthetic_model",
                "forecast_rv": forecast_rv,
                "dvol_mid_iv": dvol_level,
                "synthetic_bid_iv": bid_iv,
                "short_signal": signal,
                "entry_accepted": accepted,
                "active_positions": len(active),
                "gross_option_contracts": sum(position.contracts for position in active),
                "net_hedge_notional_usd": hedge_notional,
                "equity_btc": equity,
                "maintenance_margin_btc": maintenance,
                "margin_utilization": maintenance / equity if equity > 0 else float("inf"),
                "margin_breach": breached,
            }
        )
        previous_underlying = underlying

    if active:
        raise AssertionError("selected panel must include each position's settlement day")
    daily_frame = pd.DataFrame(daily)
    accepted = int(daily_frame["entry_accepted"].sum())
    signals = int(daily_frame["short_signal"].sum())
    skipped_capacity = signals - accepted
    return {
        "result_type": "synthetic_rolling_portfolio_envelope_not_executable_backtest",
        "parameters": {
            "horizon_days": parameters.horizon_days,
            "contracts_per_entry": parameters.contracts_per_entry,
            "max_contracts_per_btc": parameters.max_contracts_per_btc,
            "bid_iv_discount_points": parameters.bid_iv_discount_points,
            "funding_rate_hourly": parameters.funding_rate_hourly,
            "perp_taker_fee_rate": parameters.perp_taker_fee_rate,
            "initial_equity_btc": parameters.initial_equity_btc,
        },
        "coverage": {
            "first_market_at": str(panel["at"].iloc[0]),
            "last_settlement_at": str(panel["at"].iloc[-1]),
            "eligible_daily_decisions": len(decision_panel),
            "short_signals": signals,
            "accepted_entries": accepted,
            "skipped_for_book_capacity": skipped_capacity,
        },
        "summary": {
            "terminal_equity_btc": float(daily_frame["equity_btc"].iloc[-1]),
            "total_pnl_btc": float(
                daily_frame["equity_btc"].iloc[-1] - parameters.initial_equity_btc
            ),
            "max_active_positions": int(daily_frame["active_positions"].max()),
            "max_gross_option_contracts": float(daily_frame["gross_option_contracts"].max()),
            "max_margin_utilization": float(
                daily_frame["margin_utilization"].replace(float("inf"), pd.NA).max()
            ),
            "margin_breach_days": margin_breaches,
        },
        "entries": entries,
        "daily": daily,
    }


def _market_panel(
    prices: pd.DataFrame,
    dvol: pd.DataFrame,
    forecasts: pd.DataFrame,
    *,
    start_at: pd.Timestamp | None,
    end_at: pd.Timestamp | None,
) -> pd.DataFrame:
    required = {"forecast_at", "garch_corrected_rv"}
    missing = required.difference(forecasts.columns)
    if missing:
        raise ValueError(f"forecasts are missing required columns: {sorted(missing)}")
    points = forecasts.loc[:, ["forecast_at", "garch_corrected_rv"]].copy()
    points["at"] = pd.to_datetime(points.pop("forecast_at"), utc=True)
    points["garch_corrected_rv"] = pd.to_numeric(points["garch_corrected_rv"], errors="coerce")
    points = points.dropna().sort_values("at")
    if start_at is not None:
        points = points.loc[points["at"] >= _utc(start_at)]
    if end_at is not None:
        points = points.loc[points["at"] <= _utc(end_at)]
    price_panel = _available_daily_closes(prices, value_name="underlying").sort_values(
        "available_at"
    )
    dvol_panel = _available_daily_closes(dvol, value_name="dvol_points").sort_values(
        "available_at"
    )
    merged = pd.merge_asof(
        points, price_panel, left_on="at", right_on="available_at", direction="backward"
    )
    merged = pd.merge_asof(
        merged.drop(columns="available_at"),
        dvol_panel,
        left_on="at",
        right_on="available_at",
        direction="backward",
    )
    result = merged.dropna(subset=["underlying", "dvol_points"]).rename(
        columns={"dvol_points": "dvol"}
    )[["at", "underlying", "dvol", "garch_corrected_rv"]]
    # The source index is in points (e.g. 60); pricing and forecasts use decimals.
    result["dvol"] = result["dvol"] / 100
    return result


def _liability_and_delta(
    positions: list[_Position], underlying: float, dvol: float, at: pd.Timestamp
) -> tuple[float, float]:
    liability = 0.0
    delta = 0.0
    for position in positions:
        remaining_days = max((position.expiry_at - at).total_seconds() / 86_400, 0.0)
        value = _position_value(position, underlying, dvol, at)
        iv_shift = dvol - position.entry_dvol
        legs = _legs(position)
        liability += position.contracts * value
        if remaining_days > 0:
            delta += position.contracts * basket_delta_btc(
                legs,
                underlying=underlying,
                forward=underlying,
                time_years=remaining_days / 365,
                iv_shift=iv_shift,
            )
    return liability, delta


def _equity(
    cash: float,
    positions: list[_Position],
    underlying: float,
    dvol: float,
    at: pd.Timestamp,
    initial_equity_btc: float,
) -> float:
    liability = sum(
        position.contracts * _position_value(position, underlying, dvol, at)
        for position in positions
    )
    return initial_equity_btc + cash - liability


def _legs(position: _Position) -> list[dict[str, object]]:
    return [
        {"type": option_type, "strike": position.strike, "entry_iv": position.entry_iv}
        for option_type in ("call", "put")
    ]


def _position_value(position: _Position, underlying: float, dvol: float, at: pd.Timestamp) -> float:
    remaining_days = max((position.expiry_at - at).total_seconds() / 86_400, 0.0)
    return basket_value_btc(
        _legs(position),
        forward=underlying,
        time_years=remaining_days / 365,
        iv_shift=dvol - position.entry_dvol,
    )


def _straddle_value(forward: float, strike: float, iv: float, days: int) -> float:
    return sum(
        inverse_option_price(
            option_type, forward=forward, strike=strike, time_years=days / 365, volatility=iv
        )
        for option_type in ("call", "put")
    )


def _straddle_entry_fee(credit: float) -> float:
    return 2 * _option_fee(credit / 2)


def _straddle_payoff(strike: float, delivery: float) -> float:
    return sum(
        settlement_payoff_btc(option_type, strike, delivery) for option_type in ("call", "put")
    )


def _straddle_settlement_fee(strike: float, delivery: float) -> float:
    return sum(
        settlement_fee_btc(settlement_payoff_btc(option_type, strike, delivery))
        for option_type in ("call", "put")
    )


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")
