"""Sizing, margin and ruin for the delta-hedged short straddle book.

Phase 3 of the volatility pipeline plan (docs/2026-09-02-plano-pipeline-vol.md).
Layer 1: block-bootstrap the joint (BTC return, DVOL change) history and
reprice each observed quarterly contract as a daily delta-hedged book,
tracking an approximate Deribit maintenance-margin requirement.  Layer 2:
compound sequences of trades under fractional sizing with a liquidation
barrier, over a grid of sizes plus the Kelly and half-Kelly points.

Entries fill at the observed bid (crossing): the conservative floor from the
Phase 2 execution study.  Funding is the mean realized hourly rate over the
22 cached hedge windows.
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
from quant_pairs.short_straddle_bootstrap import build_joint_history, sample_block_paths
from quant_pairs.tardis_intraday import _option_fee

FUNDING_CACHE_DIR = Path("data/market/deribit/funding")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration", default="artifacts/tardis-option-spread-calibration-v1.json"
    )
    parser.add_argument(
        "--prices", default="data/market/deribit/price-bars/BTC-PERPETUAL/1D.csv.gz"
    )
    parser.add_argument("--dvol", default="data/market/deribit/volatility-index/BTC.csv.gz")
    parser.add_argument("--n-paths", type=int, default=2000)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--n-sequences", type=int, default=4000)
    parser.add_argument("--trades-per-sequence", type=int, default=26)
    parser.add_argument("--ruin-fraction", type=float, default=0.5)
    parser.add_argument("--sizes", default="0.25,0.5,1,2,4", help="contracts per BTC of equity")
    parser.add_argument("--output", default="artifacts/hedged-book-sizing-v1.json")
    args = parser.parse_args()

    calibration = _read_json(args.calibration)
    observations = pd.DataFrame(calibration["observations"])
    dvol_raw = pd.read_csv(args.dvol)
    history = build_joint_history(pd.read_csv(args.prices), dvol_raw)
    dvol_start = pd.to_datetime(dvol_raw["timestamp"], utc=True, format="mixed").min()
    funding_rate = _mean_funding_rate()

    pooled_pnl: list[np.ndarray] = []
    pooled_cum: list[np.ndarray] = []
    pooled_margin: list[np.ndarray] = []
    pooled_credit: list[np.ndarray] = []
    modeled_dates: list[str] = []
    max_horizon = 0
    per_entry: list[dict[str, object]] = []
    blocks: list[dict[str, np.ndarray]] = []
    for date, legs in observations.groupby("date", sort=True):
        if len(legs) != 2 or set(legs["option_type"]) != {"call", "put"}:
            continue
        if legs[["bid_amount", "ask_amount"]].min().min() < 0.1:
            continue
        if pd.Timestamp(legs["entry_at"].iloc[0]) < dvol_start:
            continue
        strike = float(legs["strike_usd"].iloc[0])
        dte = float(legs["dte"].iloc[0])
        forward = float(legs["parity_forward_usd"].iloc[0])
        entry_iv = float(legs["mid_iv"].mean())
        credit = float(legs["bid_btc"].sum())  # per 1 contract each leg
        fees = sum(_option_fee(float(p)) for p in legs["bid_btc"])
        horizon = max(int(np.ceil(dte)), 1)
        rng = np.random.default_rng(args.seed + int(pd.Timestamp(date).strftime("%Y%m%d")))
        paths = sample_block_paths(
            history,
            horizon=horizon,
            n_paths=args.n_paths,
            block_size=args.block_size,
            rng=rng,
        )
        result = simulate_hedged_trade_paths(
            paths,
            strike=strike,
            entry_forward=forward,
            entry_iv=entry_iv,
            dte_days=dte,
            entry_credit_btc=credit,
            entry_fees_btc=fees,
            contracts=1.0,
            funding_rate_hourly=funding_rate,
        )
        modeled_dates.append(str(date))
        max_horizon = max(max_horizon, result["cum_pnl"].shape[1])
        blocks.append(result)
        pooled_credit.append(np.full(args.n_paths, credit))
        stats = result["total_pnl"]
        per_entry.append(
            {
                "date": str(date),
                "entry_credit_btc_per_contract": credit,
                "mean_pnl_btc": float(stats.mean()),
                "p05_pnl_btc": float(np.quantile(stats, 0.05)),
                "worst_pnl_btc": float(stats.min()),
                "max_margin_btc_per_contract": float(result["margin"].max()),
            }
        )

    if not blocks:
        raise SystemExit("no eligible entries")
    for result in blocks:
        pooled_pnl.append(result["total_pnl"])
        pooled_cum.append(_pad_last(result["cum_pnl"], max_horizon))
        pooled_margin.append(_pad_zero(result["margin"], max_horizon))
    pnl = np.concatenate(pooled_pnl)
    cum = np.vstack(pooled_cum)
    margin = np.vstack(pooled_margin)
    credit_pc = np.concatenate(pooled_credit)

    kelly = kelly_fraction(pnl)
    sizes = [float(s) for s in args.sizes.split(",")]
    sizes += [kelly["kelly_contracts_per_btc"], kelly["half_kelly_contracts_per_btc"]]
    sequence_rng = np.random.default_rng(args.seed + 1)
    grid = [
        simulate_capital_sequences(
            cum,
            margin,
            credit_pc,
            contracts_per_btc=size,
            n_sequences=args.n_sequences,
            trades_per_sequence=args.trades_per_sequence,
            rng=sequence_rng,
            ruin_fraction=args.ruin_fraction,
        )
        for size in sorted(set(round(s, 6) for s in sizes if s > 0))
    ]

    payload = {
        "schema_version": 1,
        "study": "hedged short straddle book: bootstrap sizing, margin and ruin",
        "entry_fill": "crossed bid (conservative floor from the Phase 2 study)",
        "funding_rate_hourly": funding_rate,
        "n_paths_per_entry": args.n_paths,
        "modeled_dates": modeled_dates,
        "per_trade_pnl_per_contract": {
            "n": int(pnl.size),
            "mean_btc": float(pnl.mean()),
            "std_btc": float(pnl.std(ddof=1)),
            "p05_btc": float(np.quantile(pnl, 0.05)),
            "p01_btc": float(np.quantile(pnl, 0.01)),
            "worst_btc": float(pnl.min()),
            "prob_loss": float((pnl < 0).mean()),
        },
        "kelly": kelly,
        "ruin_definition": f"equity <= {args.ruin_fraction} of initial capital",
        "sizing_grid": grid,
        "per_entry": per_entry,
        "limitations": [
            "underlying proxied by the forward along bootstrap paths",
            "funding is a constant mean realized hourly rate, not a simulated series",
            "standard (non-portfolio) margin formulas; portfolio margin would be lower",
            "liquidation penalty is a declared crude constant (25% of entry credit)",
            "trades are i.i.d. resamples of the 22-entry pool; no regime persistence across trades",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "per_trade": payload["per_trade_pnl_per_contract"],
                "kelly": kelly,
                "sizing_grid": grid,
            },
            indent=2,
        )
    )


def _mean_funding_rate() -> float:
    files = sorted(FUNDING_CACHE_DIR.glob("BTC-PERPETUAL-hedge-*.csv"))
    if not files:
        raise SystemExit("no cached funding windows; run scripts/run_delta_hedged_carry.py first")
    rates = [pd.read_csv(path)["interest_1h"].mean() for path in files]
    return float(np.mean(rates))


def _pad_last(matrix: np.ndarray, width: int) -> np.ndarray:
    if matrix.shape[1] == width:
        return matrix
    pad = np.repeat(matrix[:, -1:], width - matrix.shape[1], axis=1)
    return np.hstack([matrix, pad])


def _pad_zero(matrix: np.ndarray, width: int) -> np.ndarray:
    if matrix.shape[1] == width:
        return matrix
    pad = np.zeros((matrix.shape[0], width - matrix.shape[1]))
    return np.hstack([matrix, pad])


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    main()
