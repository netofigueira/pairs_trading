"""Run the V1 daily synthetic-exit envelope for observed short BTC straddles."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.synthetic_option_backfill import (
    build_daily_straddle_marks,
    evaluate_short_exit,
    inject_gap_shock,
    settle_short_straddle,
)
from quant_pairs.tardis_intraday import _option_fee


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", default="config/experiment.synthetic-option-backfill-v1.json"
    )
    parser.add_argument(
        "--calibration", default="artifacts/tardis-option-spread-calibration-v1.json"
    )
    parser.add_argument(
        "--prices", default="data/market/deribit/price-bars/BTC-PERPETUAL/1D.csv.gz"
    )
    parser.add_argument("--dvol", default="data/market/deribit/volatility-index/BTC.csv.gz")
    parser.add_argument("--contracts", type=float, default=0.1)
    parser.add_argument("--carry", default="artifacts/tardis-carry-quarterly-v1.json")
    parser.add_argument(
        "--carry-recovered", default="artifacts/tardis-carry-quarterly-v1-min-size-failures.json"
    )
    parser.add_argument("--output", default="artifacts/synthetic-option-backfill-v1.json")
    parser.add_argument(
        "--gap-returns",
        default="0,-0.10,0.10,-0.15,0.15,-0.20,0.20",
        help="Comma-separated close-to-close gap shocks applied to the forward. "
        "0 reproduces the no-gap path. Empirical worst overnight BTC move is ~-16.5 pct.",
    )
    parser.add_argument(
        "--gap-iv-bump",
        type=float,
        default=15.0,
        help="Absolute IV points added at the gap instant for adverse tail repricing.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit repeated per-trade scenario rows from the output artifact.",
    )
    arguments = parser.parse_args()
    if arguments.contracts <= 0:
        parser.error("--contracts must be positive")
    gap_returns = [float(value) for value in arguments.gap_returns.split(",")]
    if any(not math.isfinite(value) or value <= -1 for value in gap_returns):
        parser.error("every gap return must be finite and greater than -1")
    if arguments.gap_iv_bump < 0:
        parser.error("--gap-iv-bump must be non-negative")

    experiment = _read_json(arguments.experiment)
    calibration = _read_json(arguments.calibration)
    prices = pd.read_csv(arguments.prices)
    dvol = pd.read_csv(arguments.dvol)
    observations = pd.DataFrame(calibration["observations"])
    spread_scenarios = calibration["summary"]["relative_half_spread"]
    settlements = _settlements(arguments.carry, arguments.carry_recovered)

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for date, legs in observations.groupby("date", sort=True):
        try:
            results.extend(
                _run_entry(
                    date,
                    legs,
                    prices,
                    dvol,
                    experiment,
                    spread_scenarios,
                    contracts=arguments.contracts,
                    gap_returns=gap_returns,
                    gap_iv_bump=arguments.gap_iv_bump,
                )
            )
        except ValueError as error:
            failures.append({"date": str(date), "error": str(error)})

    payload = {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"],
        "result_type": "modeled_feasibility_envelope_not_executable_backtest",
        "contracts_per_leg": arguments.contracts,
        "coverage": {
            "attempted_entries": int(observations["date"].nunique()),
            "modeled_entries": len({row["date"] for row in results}),
            "failed_entries": len(failures),
            "failures": failures,
        },
        "summaries": _summaries(results),
        "hold_to_expiry_baseline_all": _hold_to_expiry_baseline(
            observations, settlements, contracts=arguments.contracts
        ),
        "hold_to_expiry_baseline_comparable": _hold_to_expiry_baseline(
            observations,
            settlements,
            contracts=arguments.contracts,
            included_dates={row["date"] for row in results},
        ),
        "results_omitted": arguments.summary_only,
        "results": [] if arguments.summary_only else results,
        "assumptions": {
            "entry": "observed Tardis bid and displayed size >= 0.1 contract",
            "exit": "inverse Black-76 ask synthesized from empirical half-spread percentile",
            "iv_path": "observed entry mid-IV plus change in last available daily DVOL close",
            "forward_path": "daily BTC-PERPETUAL close plus entry parity-basis yield and stress",
            "daily_close_availability": "candle timestamp plus one day",
            "fees": "Deribit option fee approximation already used by carry pilot",
            "excluded": "margin, liquidation, intraday barrier breaches and dynamic hedge",
            "gap_interpretation": (
                "adversarial sensitivity: one forced shock on the ex-post most expensive "
                "daily mark of every trade; not an empirical event frequency"
            ),
            "empirical_daily_tail_counts": {
                "sample_days": 2069,
                "return_lte_minus_10pct": 13,
                "return_gte_plus_10pct": 9,
            },
        },
    }
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{serialized}\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("coverage", "summaries")}, indent=2))


def _run_entry(
    date: str,
    legs: pd.DataFrame,
    prices: pd.DataFrame,
    dvol: pd.DataFrame,
    experiment: dict,
    spread_scenarios: dict[str, float],
    *,
    contracts: float,
    gap_returns: list[float],
    gap_iv_bump: float,
) -> list[dict[str, object]]:
    if len(legs) != 2 or set(legs["option_type"]) != {"call", "put"}:
        raise ValueError("entry does not contain one call and one put")
    if legs[["bid_amount", "ask_amount"]].min().min() < contracts:
        raise ValueError("observed top-of-book size is below requested contracts")
    entry = pd.Timestamp(legs["entry_at"].iloc[0])
    expiry = entry + pd.Timedelta(days=float(legs["dte"].iloc[0]))
    strike = float(legs["strike_usd"].iloc[0])
    entry_credit = float(legs["bid_btc"].sum()) * contracts
    entry_fees = contracts * sum(_option_fee(float(price)) for price in legs["bid_btc"])
    entry_iv = float(legs["mid_iv"].mean())

    results: list[dict[str, object]] = []
    for spread_name, half_spread in spread_scenarios.items():
        for iv_stress in experiment["volatility_stress_absolute_points"]:
            for basis_stress in experiment["basis_stress_bps"]:
                marks = build_daily_straddle_marks(
                    prices,
                    dvol,
                    entry_at=entry,
                    expiry_at=expiry,
                    strike=strike,
                    entry_underlying=float(legs["underlying_perp_mid_usd"].iloc[0]),
                    entry_forward=float(legs["parity_forward_usd"].iloc[0]),
                    entry_iv=entry_iv,
                    relative_half_spread=float(half_spread),
                    iv_stress_points=float(iv_stress),
                    basis_stress_bps=float(basis_stress),
                    contracts=contracts,
                )
                for gap_return in gap_returns:
                    gapped = (
                        marks
                        if gap_return == 0
                        else inject_gap_shock(
                            marks,
                            strike=strike,
                            gap_return=gap_return,
                            iv_bump_points=gap_iv_bump,
                            contracts=contracts,
                        )
                    )
                    for rule in experiment["exit_rules"]:
                        exit_result = evaluate_short_exit(
                            gapped,
                            entry_credit_btc=entry_credit,
                            profit_target=float(rule["profit_target"]),
                            stop_multiple=float(rule["stop_multiple"]),
                            exit_dte=float(rule["exit_dte"]),
                        )
                        net_pnl = (
                            float(exit_result["pnl_before_entry_fee_btc"]) - entry_fees
                        )
                        results.append(
                            {
                                "date": str(date),
                                "rule": rule["name"],
                                "spread_scenario": spread_name,
                                "iv_stress_points": iv_stress,
                                "basis_stress_bps": basis_stress,
                                "gap_return": gap_return,
                                "entry_credit_btc": entry_credit,
                                "entry_fees_btc": entry_fees,
                                **exit_result,
                                "net_pnl_btc": net_pnl,
                                "return_on_credit": net_pnl / entry_credit,
                            }
                        )
    return results


def _summaries(results: list[dict[str, object]]) -> list[dict[str, object]]:
    if not results:
        return []
    frame = pd.DataFrame(results)
    rows: list[dict[str, object]] = []
    keys = ["rule", "spread_scenario", "iv_stress_points", "basis_stress_bps", "gap_return"]
    for values, group in frame.groupby(keys, sort=True):
        returns = group["return_on_credit"].to_numpy(dtype=float)
        pnl = group["net_pnl_btc"].to_numpy(dtype=float)
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "observations": len(group),
                "positive": int((pnl > 0).sum()),
                "total_pnl_btc": float(pnl.sum()),
                "mean_return_on_credit": float(np.mean(returns)),
                "median_return_on_credit": float(np.median(returns)),
                "worst_return_on_credit": float(np.min(returns)),
                "exit_triggers": dict(Counter(group["exit_trigger"])),
            }
        )
    return rows


def _hold_to_expiry_baseline(
    observations: pd.DataFrame,
    settlements: dict[str, float],
    *,
    contracts: float,
    included_dates: set[str] | None = None,
) -> dict[str, object]:
    rows = []
    for date, legs in observations.groupby("date", sort=True):
        if included_dates is not None and date not in included_dates:
            continue
        if date not in settlements or legs[["bid_amount", "ask_amount"]].min().min() < contracts:
            continue
        credit = float(legs["bid_btc"].sum()) * contracts
        entry_fees = contracts * sum(_option_fee(float(price)) for price in legs["bid_btc"])
        pnl = settle_short_straddle(
            entry_credit_btc=credit,
            entry_fees_btc=entry_fees,
            settlement_payoff_per_contract_btc=settlements[date],
            contracts=contracts,
        )
        rows.append({"date": date, "net_pnl_btc": pnl, "return_on_credit": pnl / credit})
    frame = pd.DataFrame(rows)
    return {
        "observations": len(frame),
        "positive": int((frame["net_pnl_btc"] > 0).sum()),
        "total_pnl_btc": float(frame["net_pnl_btc"].sum()),
        "mean_return_on_credit": float(frame["return_on_credit"].mean()),
        "median_return_on_credit": float(frame["return_on_credit"].median()),
        "worst_return_on_credit": float(frame["return_on_credit"].min()),
        "results": rows,
    }


def _settlements(*paths: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for path in paths:
        for row in _read_json(path):
            if row.get("status") == "failed":
                continue
            date = str(row["entry_at"])[:10]
            contracts = float(row["contracts_per_leg"])
            values[date] = float(row["settlement_payoff_btc"]) / contracts
    return values


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
