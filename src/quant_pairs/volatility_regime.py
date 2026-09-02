"""Frozen executable long/short/flat volatility-regime gate."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd
from scipy.stats import t

from quant_pairs.tardis_intraday import _option_fee


def classify_volatility_regime(
    forecast_variance: float,
    *,
    bid_ivs: Iterable[float],
    ask_ivs: Iterable[float],
) -> str:
    """Classify against executable IV boundaries, with no fitted threshold."""

    bid = [float(value) for value in bid_ivs]
    ask = [float(value) for value in ask_ivs]
    if forecast_variance <= 0 or not bid or len(bid) != len(ask):
        raise ValueError("forecast and paired IV observations must be positive")
    if any(value <= 0 for value in (*bid, *ask)):
        raise ValueError("IV observations must be positive")
    bid_variance = sum(value**2 for value in bid) / len(bid)
    ask_variance = sum(value**2 for value in ask) / len(ask)
    if bid_variance > ask_variance:
        raise ValueError("bid IV cannot exceed ask IV")
    if forecast_variance > ask_variance:
        return "long"
    if forecast_variance < bid_variance:
        return "short"
    return "flat"


def build_economic_gate(
    forecasts: list[dict[str, object]],
    option_observations: list[dict[str, object]],
    carry_rows: list[dict[str, object]],
    *,
    evaluation_start: str = "2021-04-01T00:00:00Z",
    contracts: float = 0.1,
    minimum_actionable: int = 12,
    minimum_each_side: int = 3,
) -> dict[str, object]:
    """Apply the frozen rule to observed entry books and official settlement.

    Forecast selection is strictly as-of. Carry outcomes are used only after
    the action has been assigned and are normalized to a common position size.
    """

    if contracts <= 0 or minimum_actionable <= 0 or minimum_each_side <= 0:
        raise ValueError("gate controls must be positive")
    start = pd.Timestamp(evaluation_start)
    if start.tzinfo is None:
        raise ValueError("evaluation_start must be timezone-aware")

    forecast_frame = pd.DataFrame(forecasts)
    if forecast_frame.empty:
        raise ValueError("forecasts cannot be empty")
    forecast_frame["forecast_at"] = pd.to_datetime(forecast_frame["forecast_at"], utc=True)
    forecast_frame = forecast_frame.dropna(subset=["garch_corrected_rv"])
    forecast_frame = forecast_frame.sort_values("forecast_at")

    options = pd.DataFrame(option_observations)
    options["entry_at"] = pd.to_datetime(options["entry_at"], utc=True)
    options = options.loc[options["entry_at"] >= start].copy()
    settlements = _settlements_by_date(carry_rows)
    points: list[dict[str, object]] = []
    for entry_at, legs in options.groupby("entry_at", sort=True):
        date = entry_at.strftime("%Y-%m-%d")
        prior = forecast_frame.loc[forecast_frame["forecast_at"] <= entry_at]
        if prior.empty:
            points.append({"entry_at": str(entry_at), "status": "forecast_unavailable"})
            continue
        if date not in settlements:
            points.append({"entry_at": str(entry_at), "status": "settlement_unavailable"})
            continue
        forecast = prior.iloc[-1]
        forecast_variance = float(forecast["garch_corrected_rv"]) ** 2
        bid_ivs = legs["bid_iv"].astype(float).tolist()
        ask_ivs = legs["ask_iv"].astype(float).tolist()
        action = classify_volatility_regime(forecast_variance, bid_ivs=bid_ivs, ask_ivs=ask_ivs)
        settlement = settlements[date]
        entry_ask = float(legs["ask_btc"].sum())
        entry_bid = float(legs["bid_btc"].sum())
        long_pnl = (
            settlement["payoff"]
            - entry_ask
            - sum(_option_fee(price) for price in legs["ask_btc"].astype(float))
            - settlement["fees"]
        ) * contracts
        short_pnl = (
            entry_bid
            - sum(_option_fee(price) for price in legs["bid_btc"].astype(float))
            - settlement["payoff"]
            - settlement["fees"]
        ) * contracts
        selected_pnl = {"long": long_pnl, "short": short_pnl, "flat": 0.0}[action]
        opposite_pnl = {"long": short_pnl, "short": long_pnl, "flat": 0.0}[action]
        points.append(
            {
                "entry_at": str(entry_at),
                "status": "evaluated",
                "forecast_at": str(forecast["forecast_at"]),
                "forecast_rv": math.sqrt(forecast_variance),
                "bid_iv": math.sqrt(sum(value**2 for value in bid_ivs) / len(bid_ivs)),
                "ask_iv": math.sqrt(sum(value**2 for value in ask_ivs) / len(ask_ivs)),
                "action": action,
                "long_pnl_btc": long_pnl,
                "short_pnl_btc": short_pnl,
                "selected_pnl_btc": selected_pnl,
                "selected_advantage_btc": selected_pnl - opposite_pnl,
            }
        )

    evaluated = [point for point in points if point["status"] == "evaluated"]
    actions = {
        name: sum(point["action"] == name for point in evaluated)
        for name in ("long", "short", "flat")
    }
    selected = [float(point["selected_pnl_btc"]) for point in evaluated]
    advantages = [float(point["selected_advantage_btc"]) for point in evaluated]
    selected_summary = _pnl_summary(selected)
    always_long = _pnl_summary([float(point["long_pnl_btc"]) for point in evaluated])
    always_short = _pnl_summary([float(point["short_pnl_btc"]) for point in evaluated])
    by_action = {
        action: _pnl_summary(
            [float(point["selected_pnl_btc"]) for point in evaluated if point["action"] == action]
        )
        for action in ("long", "short", "flat")
    }
    enough_sides = actions["long"] >= minimum_each_side and actions["short"] >= minimum_each_side
    checks = {
        "minimum_actionable": actions["long"] + actions["short"] >= minimum_actionable,
        "both_sides_represented": enough_sides,
        "positive_selected_total": selected_summary["total_pnl_btc"] > 0,
        "positive_pnl_in_each_directional_regime": all(
            by_action[action]["total_pnl_btc"] > 0 for action in ("long", "short")
        ),
        "positive_mean_selected_advantage": bool(advantages)
        and sum(advantages) / len(advantages) > 0,
    }
    return {
        "rule": {
            "model": "causally bias-corrected GARCH(1,1), 14-day variance",
            "long": "forecast variance > mean executable ask-IV variance",
            "short": "forecast variance < mean executable bid-IV variance",
            "flat": "forecast variance inside the executable IV spread",
            "threshold_fitted": False,
        },
        "coverage": {
            "eligible_entries": len(points),
            "evaluated": len(evaluated),
            "unavailable": len(points) - len(evaluated),
            "fixed_contracts_per_leg": contracts,
        },
        "actions": actions,
        "by_action": by_action,
        "selected": selected_summary,
        "always_long": always_long,
        "always_short": always_short,
        "promotion_checks": checks,
        "promote_to_monthly": all(checks.values()),
        "points": points,
    }


def _settlements_by_date(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("status") == "failed":
            continue
        contracts = float(row["contracts_per_leg"])
        date = str(row["entry_at"])[:10]
        result[date] = {
            "payoff": float(row["settlement_payoff_btc"]) / contracts,
            "fees": float(row["settlement_fees_btc"]) / contracts,
        }
    return result


def _pnl_summary(values: list[float], confidence: float = 0.95) -> dict[str, float | int]:
    if not values:
        return {"observations": 0, "total_pnl_btc": 0.0, "mean_pnl_btc": 0.0}
    series = pd.Series(values, dtype=float)
    mean = float(series.mean())
    if len(series) > 1:
        error = float(series.std(ddof=1) / math.sqrt(len(series)))
        critical = float(t.ppf(0.5 + confidence / 2, df=len(series) - 1))
        low, high = mean - critical * error, mean + critical * error
    else:
        low = high = mean
    return {
        "observations": len(series),
        "total_pnl_btc": float(series.sum()),
        "mean_pnl_btc": mean,
        "mean_ci_low": low,
        "mean_ci_high": high,
        "positive": int((series > 0).sum()),
    }
