"""Regime-conditioned bootstrap of the hedged short straddle book.

Pre-declared design: config/experiment.conditioned-bootstrap-v1.json.  Same
hedged-book repricing as Phase 3, but blocks may only start on days whose
previous available DVOL close lies within a tolerance of the entry's DVOL, so
the null preserves the volatility regime the strategy claims to read.  The
primary question is whether the frozen gate's *short* entries keep a positive
pooled mean under this null; sizing is re-run on the gate-short pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_pairs.hedged_book_bootstrap import (
    kelly_fraction,
    simulate_capital_sequences,
    simulate_hedged_trade_paths,
)
from quant_pairs.short_straddle_bootstrap import (
    build_joint_history_with_levels,
    sample_conditioned_block_paths,
)
from quant_pairs.synthetic_option_backfill import _available_daily_closes, _last_available
from quant_pairs.tardis_intraday import _option_fee

FUNDING_CACHE_DIR = Path("data/market/deribit/funding")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="config/experiment.conditioned-bootstrap-v1.json")
    parser.add_argument(
        "--calibration", default="artifacts/tardis-option-spread-calibration-v1.json"
    )
    parser.add_argument("--gate", default="artifacts/volatility-regime-gate-v1.json")
    parser.add_argument("--unconditional", default="artifacts/hedged-book-sizing-v1.json")
    parser.add_argument(
        "--prices", default="data/market/deribit/price-bars/BTC-PERPETUAL/1D.csv.gz"
    )
    parser.add_argument("--dvol", default="data/market/deribit/volatility-index/BTC.csv.gz")
    parser.add_argument("--n-sequences", type=int, default=4000)
    parser.add_argument("--trades-per-sequence", type=int, default=26)
    parser.add_argument("--ruin-fraction", type=float, default=0.5)
    parser.add_argument("--sizes", default="0.25,0.5,1,2")
    parser.add_argument("--output", default="artifacts/conditioned-bootstrap-v1.json")
    args = parser.parse_args()

    experiment = _read_json(args.experiment)
    conditioning = experiment["conditioning"]
    resampling = experiment["resampling"]
    observations = pd.DataFrame(_read_json(args.calibration)["observations"])
    gate_actions = {
        str(point["entry_at"])[:10]: point.get("action")
        for point in _read_json(args.gate)["gate"]["points"]
    }
    unconditional = {
        entry["date"]: entry["mean_pnl_btc"]
        for entry in _read_json(args.unconditional)["per_entry"]
    }

    prices = pd.read_csv(args.prices)
    dvol_raw = pd.read_csv(args.dvol)
    history, levels = build_joint_history_with_levels(prices, dvol_raw)
    dvol_panel = _available_daily_closes(dvol_raw, value_name="dvol_points")
    dvol_start = pd.to_datetime(dvol_raw["timestamp"], utc=True, format="mixed").min()
    funding_rate = _mean_funding_rate()

    per_entry: list[dict[str, object]] = []
    max_horizon = 0
    blocks: list[tuple[str, dict[str, np.ndarray], float]] = []
    for date, legs in observations.groupby("date", sort=True):
        if len(legs) != 2 or set(legs["option_type"]) != {"call", "put"}:
            continue
        if legs[["bid_amount", "ask_amount"]].min().min() < 0.1:
            continue
        entry_at = pd.Timestamp(legs["entry_at"].iloc[0])
        if entry_at < dvol_start:
            continue
        entry_dvol = _last_available(dvol_panel, entry_at, "dvol_points")
        strike = float(legs["strike_usd"].iloc[0])
        dte = float(legs["dte"].iloc[0])
        forward = float(legs["parity_forward_usd"].iloc[0])
        entry_iv = float(legs["mid_iv"].mean())
        credit = float(legs["bid_btc"].sum())
        fees = sum(_option_fee(float(p)) for p in legs["bid_btc"])
        horizon = max(int(np.ceil(dte)), 1)
        rng = np.random.default_rng(
            int(resampling["seed"]) + int(pd.Timestamp(date).strftime("%Y%m%d"))
        )
        paths, info = sample_conditioned_block_paths(
            history,
            levels,
            entry_dvol_points=entry_dvol,
            horizon=horizon,
            n_paths=int(resampling["n_paths_per_entry"]),
            block_size=int(resampling["block_size_days"]),
            rng=rng,
            tolerance_points=float(conditioning["tolerance_vol_points"]),
            widening_step_points=float(conditioning["widening_step_vol_points"]),
            min_starts=int(conditioning["min_eligible_starts"]),
        )
        result = simulate_hedged_trade_paths(
            paths,
            strike=strike,
            entry_forward=forward,
            entry_iv=entry_iv,
            dte_days=dte,
            entry_credit_btc=credit,
            entry_fees_btc=fees,
            funding_rate_hourly=funding_rate,
        )
        action = gate_actions.get(str(date)) or "no_forecast"
        max_horizon = max(max_horizon, result["cum_pnl"].shape[1])
        blocks.append((action, result, credit))
        realized_vol = float(
            np.std(paths[:, :, 0], ddof=1) * np.sqrt(365)
        )  # pooled daily-return vol of the conditioned sample
        per_entry.append(
            {
                "date": str(date),
                "gate_action": action,
                "entry_iv": entry_iv,
                "entry_dvol_points": info["entry_dvol_points"],
                "tolerance_points_used": info["tolerance_points_used"],
                "eligible_starts": int(info["eligible_starts"]),
                "conditioned_sample_vol_ann": realized_vol,
                "mean_pnl_btc": float(result["total_pnl"].mean()),
                "p05_pnl_btc": float(np.quantile(result["total_pnl"], 0.05)),
                "worst_pnl_btc": float(result["total_pnl"].min()),
                "unconditional_mean_pnl_btc": unconditional.get(str(date)),
            }
        )

    if not blocks:
        raise SystemExit("no eligible entries")

    def _pool(actions: set[str]) -> dict[str, np.ndarray] | None:
        selected = [(result, credit) for action, result, credit in blocks if action in actions]
        if not selected:
            return None
        return {
            "pnl": np.concatenate([r["total_pnl"] for r, _ in selected]),
            "cum": np.vstack([_pad_last(r["cum_pnl"], max_horizon) for r, _ in selected]),
            "margin": np.vstack([_pad_zero(r["margin"], max_horizon) for r, _ in selected]),
            "credit": np.concatenate([np.full(len(r["total_pnl"]), c) for r, c in selected]),
        }

    summaries: dict[str, object] = {}
    for name, actions in (
        ("all", {"short", "long", "no_forecast"}),
        ("gate_short", {"short"}),
        ("gate_long", {"long"}),
    ):
        pool = _pool(actions)
        if pool is None:
            continue
        pnl = pool["pnl"]
        summaries[name] = {
            "n_paths": int(pnl.size),
            "mean_btc": float(pnl.mean()),
            "std_btc": float(pnl.std(ddof=1)),
            "prob_loss": float((pnl < 0).mean()),
            "p05_btc": float(np.quantile(pnl, 0.05)),
            "worst_btc": float(pnl.min()),
        }

    short_pool = _pool({"short"})
    kelly = kelly_fraction(short_pool["pnl"]) if short_pool is not None else None
    grid = []
    if short_pool is not None and kelly is not None:
        sizes = [float(s) for s in args.sizes.split(",")]
        sizes += [
            kelly["kelly_contracts_per_btc"],
            kelly["half_kelly_contracts_per_btc"],
        ]
        sequence_rng = np.random.default_rng(int(resampling["seed"]) + 1)
        grid = [
            simulate_capital_sequences(
                short_pool["cum"],
                short_pool["margin"],
                short_pool["credit"],
                contracts_per_btc=size,
                n_sequences=args.n_sequences,
                trades_per_sequence=args.trades_per_sequence,
                rng=sequence_rng,
                ruin_fraction=args.ruin_fraction,
            )
            for size in sorted(set(round(s, 6) for s in sizes if s > 0))
        ]

    success = bool(summaries.get("gate_short", {}).get("mean_btc", -1.0) > 0)
    payload = {
        "schema_version": 1,
        "study": "regime-conditioned bootstrap of the hedged short straddle book",
        "experiment": experiment["experiment_id"],
        "funding_rate_hourly": funding_rate,
        "success_criterion": experiment["evaluation"]["success_criterion"],
        "gate_short_positive_under_conditioned_null": success,
        "pools": summaries,
        "kelly_gate_short": kelly,
        "sizing_grid_gate_short": grid,
        "per_entry": per_entry,
        "limitations": [
            "conditioning uses DVOL at block starts; within-block dynamics may exit the regime",
            "same approximations as Phase 3 (forward proxy, constant funding, std margin)",
            "gate-short pool has 6 entries; many paths but little entry-level diversity",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "gate_short_positive": success,
                "pools": summaries,
                "kelly_gate_short": kelly,
                "sizing_grid_gate_short": grid,
                "per_entry": [
                    {
                        k: e[k]
                        for k in (
                            "date",
                            "gate_action",
                            "entry_iv",
                            "conditioned_sample_vol_ann",
                            "mean_pnl_btc",
                            "unconditional_mean_pnl_btc",
                        )
                    }
                    for e in per_entry
                ],
            },
            indent=2,
        )
    )


def _mean_funding_rate() -> float:
    files = sorted(FUNDING_CACHE_DIR.glob("BTC-PERPETUAL-hedge-*.csv"))
    if not files:
        raise SystemExit("no cached funding windows; run scripts/run_delta_hedged_carry.py first")
    return float(np.mean([pd.read_csv(path)["interest_1h"].mean() for path in files]))


def _pad_last(matrix: np.ndarray, width: int) -> np.ndarray:
    if matrix.shape[1] == width:
        return matrix
    return np.hstack([matrix, np.repeat(matrix[:, -1:], width - matrix.shape[1], axis=1)])


def _pad_zero(matrix: np.ndarray, width: int) -> np.ndarray:
    if matrix.shape[1] == width:
        return matrix
    return np.hstack([matrix, np.zeros((matrix.shape[0], width - matrix.shape[1]))])


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
