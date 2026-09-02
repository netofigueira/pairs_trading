"""Block-bootstrap loss distribution for observed short BTC straddles.

Resamples the joint (BTC return, DVOL change) path from history and reprices
each observed contract over its real DTE.  Reports a conditional-per-trade loss
distribution -- NOT probability of ruin, which needs capital, sizing, margin and
a liquidation barrier not modeled here.  The three pre-declared exit rules are
compared against hold-to-expiry; no new rule is added after seeing results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.short_straddle_bootstrap import (
    build_joint_history,
    loss_statistics,
    sample_block_paths,
    simulate_trade_losses,
)
from quant_pairs.tardis_intraday import _option_fee

HOLD_RULE = {
    "name": "hold_to_expiry",
    "profit_target": None,
    "stop_multiple": None,
    "exit_dte": 0.0,
}


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
    parser.add_argument("--n-paths", type=int, default=10_000)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--spread-scenarios", default="p50,p90,p95")
    parser.add_argument("--output", default="artifacts/short-straddle-bootstrap-v1.json")
    args = parser.parse_args()
    if args.contracts <= 0 or args.n_paths <= 0 or args.block_size <= 0:
        parser.error("contracts, n-paths and block-size must be positive")

    experiment = _read_json(args.experiment)
    calibration = _read_json(args.calibration)
    observations = pd.DataFrame(calibration["observations"])
    all_spreads = calibration["summary"]["relative_half_spread"]
    spreads = {name: all_spreads[name] for name in args.spread_scenarios.split(",")}

    dvol_raw = pd.read_csv(args.dvol)
    history = build_joint_history(pd.read_csv(args.prices), dvol_raw)
    # Match the daily-envelope gate: exclude entries before DVOL existed, so the
    # sample stays 22/26 and no entry uses volatility data it could not observe.
    dvol_start = pd.to_datetime(dvol_raw["timestamp"], utc=True, format="mixed").min()
    rules = list(experiment["exit_rules"]) + [HOLD_RULE]

    # Pool paths per (rule, spread) across all entries so the loss distribution
    # reflects the full trade population, not a single contract.
    pooled: dict[tuple[str, str], list[np.ndarray]] = {}
    per_entry: list[dict[str, object]] = []
    modeled_dates: list[str] = []
    for date, legs in observations.groupby("date", sort=True):
        if len(legs) != 2 or set(legs["option_type"]) != {"call", "put"}:
            continue
        if legs[["bid_amount", "ask_amount"]].min().min() < args.contracts:
            continue
        if pd.Timestamp(legs["entry_at"].iloc[0]) < dvol_start:
            continue  # DVOL did not exist yet at entry; excluded like the envelope
        strike = float(legs["strike_usd"].iloc[0])
        dte = float(legs["dte"].iloc[0])
        forward = float(legs["parity_forward_usd"].iloc[0])
        entry_iv = float(legs["mid_iv"].mean())
        credit = float(legs["bid_btc"].sum()) * args.contracts
        fees = args.contracts * sum(_option_fee(float(p)) for p in legs["bid_btc"])
        horizon = max(int(np.ceil(dte)), 1)
        # Python's hash() is randomized between processes. YYYYMMDD provides a
        # stable per-entry stream while keeping rules/spreads on common paths.
        date_seed_offset = int(pd.Timestamp(date).strftime("%Y%m%d"))
        rng = np.random.default_rng(args.seed + date_seed_offset)
        paths = sample_block_paths(
            history, horizon=horizon, n_paths=args.n_paths,
            block_size=args.block_size, rng=rng,
        )
        modeled_dates.append(str(date))
        for spread_name, half_spread in spreads.items():
            for rule in rules:
                pnl = simulate_trade_losses(
                    paths,
                    entry_credit_btc=credit,
                    entry_fees_btc=fees,
                    strike=strike,
                    entry_forward=forward,
                    entry_iv=entry_iv,
                    dte_days=dte,
                    relative_half_spread=float(half_spread),
                    profit_target=_optional_float(rule["profit_target"]),
                    stop_multiple=_optional_float(rule["stop_multiple"]),
                    exit_dte=float(rule["exit_dte"]),
                    contracts=args.contracts,
                )
                key = (rule["name"], spread_name)
                pooled.setdefault(key, []).append(pnl / credit)  # normalize to credit units
                per_entry.append(
                    {
                        "date": str(date), "rule": rule["name"], "spread_scenario": spread_name,
                        "entry_credit_btc": credit,
                        **{k: v for k, v in loss_statistics(pnl, entry_credit_btc=credit).items()},
                    }
                )

    summaries = []
    for (rule_name, spread_name), samples in sorted(pooled.items()):
        pooled_return = np.concatenate(samples)  # in credit units
        stats = loss_statistics(pooled_return, entry_credit_btc=1.0)
        # Pooled samples were normalized before concatenation. Do not expose
        # dimensionless values under misleading *_btc field names.
        pooled_stats = {
            key: value
            for key, value in stats.items()
            if not key.endswith("_btc")
        }
        summaries.append(
            {"rule": rule_name, "spread_scenario": spread_name, **pooled_stats}
        )

    payload = {
        "schema_version": 1,
        "experiment_id": experiment["experiment_id"] + "-bootstrap",
        "result_type": "conditional_per_trade_loss_distribution_not_ruin",
        "method": "moving-block bootstrap of joint (BTC return, DVOL change)",
        "contracts_per_leg": args.contracts,
        "n_paths": args.n_paths, "block_size": args.block_size, "seed": args.seed,
        "coverage": {"modeled_entries": len(modeled_dates), "dates": modeled_dates},
        "pooled_summaries": summaries,
        "per_entry": per_entry,
        "caveats": [
            "Loss units are multiples of entry credit; VaR/ES on the loss side.",
            "NOT probability of ruin: no capital, sizing, margin or liquidation barrier.",
            "Rules are pre-declared; hold-to-expiry is the baseline. No rule added ex post.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    output.write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps(
        {"coverage": payload["coverage"]["modeled_entries"], "pooled_summaries": summaries},
        indent=2,
    ))


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


if __name__ == "__main__":
    main()
