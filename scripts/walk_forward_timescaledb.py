"""Run a leakage-safe rolling walk-forward evaluation from TimescaleDB."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg
from screen_timescaledb import _load_prices

from quant_pairs.backtest import BacktestConfig
from quant_pairs.walk_forward import WalkForwardConfig, run_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="config/universe.crypto-usdm-v1.json")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--formation-bars", type=int, default=4_320)
    parser.add_argument("--trade-bars", type=int, default=168)
    parser.add_argument("--step-bars", type=int, default=168)
    parser.add_argument("--max-folds", type=int, default=24)
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--stop-z", type=float, default=4.0)
    parser.add_argument("--max-holding-bars", type=int, default=72)
    parser.add_argument("--taker-fee-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=1.0)
    args = parser.parse_args()
    universe = json.loads(Path(args.universe).read_text())
    wf = WalkForwardConfig(
        formation_bars=args.formation_bars, trade_bars=args.trade_bars, step_bars=args.step_bars
    )
    if args.max_folds <= 0:
        raise SystemExit("max-folds must be positive")
    # Enough common candles for the requested number of weekly folds.  The
    # query is bounded deliberately: it cannot silently pull an unknown range.
    bars = wf.formation_bars + wf.trade_bars + ((args.max_folds - 1) * wf.step_bars)
    with psycopg.connect(os.environ["QUANT_PAIRS_DATABASE_URL"]) as connection:
        prices = _load_prices(connection, universe["symbols"], args.interval, bars)
    result = run_walk_forward(
        prices,
        BacktestConfig(
            entry_z=args.entry_z,
            exit_z=args.exit_z,
            stop_z=args.stop_z,
            max_holding_bars=args.max_holding_bars,
            taker_fee_bps=args.taker_fee_bps,
            slippage_bps=args.slippage_bps,
        ),
        wf,
    )
    print(json.dumps(result.metrics(), default=str, sort_keys=True))
    print(result.folds.to_string(index=False))


if __name__ == "__main__":
    main()
