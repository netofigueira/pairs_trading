"""Compact, versionable payload for the volatility research dashboard."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from quant_pairs.dvol import build_iv_rv_panel, non_overlapping


def build_volatility_report(
    dvol: pd.DataFrame,
    prices: pd.DataFrame,
    carry_rows: list[dict[str, object]],
    recovered_rows: list[dict[str, object]],
    *,
    horizon_days: int = 30,
) -> dict[str, object]:
    """Build chart-ready research data without embedding raw market files."""

    daily = build_iv_rv_panel(dvol, prices, horizon_days=horizon_days)
    independent = non_overlapping(daily, horizon_days=horizon_days)
    primary_carry = [row for row in carry_rows if row.get("status") != "failed"]
    failures = [row for row in carry_rows if row.get("status") == "failed"]
    carry = [*_carry_points(primary_carry), *_carry_points(recovered_rows)]
    carry.sort(key=lambda row: str(row["entry_date"]))
    returns = pd.Series([float(row["return_on_premium"]) for row in carry], dtype=float)

    return {
        "schema_version": 1,
        "study": {
            "title": "Pesquisa de volatilidade BTC",
            "status": "research_only",
            "dvol_horizon_days": horizon_days,
            "carry_target_dte": 14,
            "notes": [
                "DVOL versus RV is a calibration diagnostic, not executable option P&L.",
                "Carry entries occur only on the first day of each quarter.",
                "The carry pilot is an unhedged long straddle held to expiry.",
                "No result authorizes real-money trading or mechanically approves short-vol.",
            ],
        },
        "calibration": {
            "daily_summary": _calibration_summary(daily),
            "independent_summary": _calibration_summary(independent),
            "daily": _calibration_points(daily),
            "independent": _calibration_points(independent),
        },
        "carry": {
            "summary": {
                "attempted": len(carry_rows),
                "one_contract_fills": len(primary_carry),
                "one_contract_failures": len(failures),
                "minimum_size_recoveries": len(recovered_rows),
                "comparable_observations": len(carry),
                "positive": int((returns > 0).sum()),
                "positive_share": _float((returns > 0).mean()),
                "mean_return_on_premium": _float(returns.mean()),
                "median_return_on_premium": _float(returns.median()),
                "best_return_on_premium": _float(returns.max()),
                "worst_return_on_premium": _float(returns.min()),
            },
            "points": carry,
            "one_contract_failures": failures,
        },
    }


def _calibration_summary(panel: pd.DataFrame) -> dict[str, float | int | None]:
    correlation = panel["iv"].corr(panel["forward_rv"]) if len(panel) > 1 else None
    return {
        "observations": len(panel),
        "mean_iv": _float(panel["iv"].mean()),
        "mean_forward_rv": _float(panel["forward_rv"].mean()),
        "mean_iv_minus_rv": _float(panel["iv_minus_rv"].mean()),
        "iv_above_rv_share": _float((panel["iv_minus_rv"] > 0).mean()),
        "iv_rv_correlation": _float(correlation),
    }


def _calibration_points(panel: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "date": row.timestamp.date().isoformat(),
            "iv": float(row.iv),
            "forward_rv": float(row.forward_rv),
            "iv_minus_rv": float(row.iv_minus_rv),
            "variance_premium": float(row.iv**2 - row.forward_rv**2),
        }
        for row in panel.itertuples(index=False)
    ]


def _carry_points(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    points = []
    for row in rows:
        premium = float(row["entry_premium_btc"])
        legs = list(row["legs"])
        points.append(
            {
                "entry_date": str(row["entry_at"])[:10],
                "expiry_date": str(row["expiry_at"])[:10],
                "days_held": float(row["days_held"]),
                "strike": float(legs[0]["strike"]),
                "contracts_per_leg": float(row["contracts_per_leg"]),
                "entry_premium_btc": premium,
                "settlement_payoff_btc": float(row["settlement_payoff_btc"]),
                "fees_btc": float(row["option_entry_fees_btc"])
                + float(row["settlement_fees_btc"]),
                "net_pnl_btc": float(row["net_unhedged_pnl_btc"]),
                "return_on_premium": float(row["net_unhedged_pnl_btc"]) / premium,
            }
        )
    return points


def _float(value: object) -> float | None:
    return None if pd.isna(value) else float(value)
