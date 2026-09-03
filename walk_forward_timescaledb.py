"""Evaluate declared intraday walk-forward variants, reserving a final holdout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import psycopg
from screen_timescaledb import _load_prices

from quant_pairs.backtest import BacktestConfig
from quant_pairs.walk_forward import WalkForwardConfig, run_walk_forward

DEFAULT_EXPERIMENT = "config/experiment.crypto-usdm-intraday-v1.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--history-bars",
        type=int,
        default=8_760,
        help="maximum recent common bars to load; default is 365 days at 1h",
    )
    parser.add_argument(
        "--formation-bars",
        type=int,
        nargs="+",
        help="optional override for the declared formation-window variants",
    )
    args = parser.parse_args()
    if args.history_bars <= 0:
        raise SystemExit("history-bars must be positive")

    spec = json.loads(Path(args.experiment).read_text())
    formation_variants = args.formation_bars or spec["formation_bars"]
    _validate_spec(spec, formation_variants)
    universe = json.loads(Path(spec["universe"]).read_text())

    with psycopg.connect(os.environ["QUANT_PAIRS_DATABASE_URL"]) as connection:
        prices = _load_prices(
            connection,
            universe["symbols"],
            spec["interval"],
            args.history_bars,
            require_bars=False,
        )
        funding_by_symbol = (
            _load_funding(connection, universe["symbols"], prices.index.min(), prices.index.max())
            if spec.get("include_funding", False)
            else None
        )
    if len(prices) <= spec["holdout_bars"]:
        raise SystemExit("not enough common bars after reserving the final holdout")

    research_prices = prices.iloc[: -spec["holdout_bars"]]
    holdout = prices.iloc[-spec["holdout_bars"] :]
    signal_variants = spec.get("signal_variants", [{"name": "baseline"}])
    reports = []
    for formation_bars in formation_variants:
        config = WalkForwardConfig(
            formation_bars=formation_bars,
            trade_bars=spec["trade_bars"],
            step_bars=spec["step_bars"],
            fdr_alpha=spec["fdr_alpha"],
            min_half_life_bars=spec["min_half_life_bars"],
            max_half_life_bars=spec["max_half_life_bars"],
            portfolio_matching=spec.get("portfolio_matching", False),
        )
        if len(research_prices) < config.formation_bars + config.trade_bars:
            reports.append(
                {
                    "formation_bars": formation_bars,
                    "status": "skipped_insufficient_research_history",
                    "research_bars": len(research_prices),
                }
            )
            continue
        for variant in signal_variants:
            name = variant.get("name")
            if not isinstance(name, str) or not name:
                raise SystemExit("every signal variant needs a non-empty name")
            execution = BacktestConfig(
                **{**spec["execution"], **_execution_overrides(variant)}
            )
            result = run_walk_forward(
                research_prices,
                execution,
                config,
                funding_by_symbol=funding_by_symbol,
            )
            reports.append(
                {
                    "formation_bars": formation_bars,
                    "formation_days": formation_bars / 24,
                    "signal_variant": name,
                    "status": "evaluated",
                    "research_start": str(research_prices.index.min()),
                    "research_end": str(research_prices.index.max()),
                    **result.metrics(),
                }
            )

    print(
        json.dumps(
            {
                "experiment": spec["name"],
                "interval": spec["interval"],
                "available_common_bars": len(prices),
                "include_funding": funding_by_symbol is not None,
                "holdout": {
                    "bars": len(holdout),
                    "start": str(holdout.index.min()),
                    "end": str(holdout.index.max()),
                    "status": "reserved_not_evaluated",
                },
                "reports": reports,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _validate_spec(spec: dict, formation_variants: list[int]) -> None:
    required = {
        "name", "universe", "interval", "holdout_bars", "trade_bars", "step_bars",
        "fdr_alpha", "min_half_life_bars", "max_half_life_bars", "execution",
    }
    missing = required.difference(spec)
    if missing:
        raise SystemExit(f"experiment is missing fields: {sorted(missing)}")
    if not formation_variants or any(value < 90 for value in formation_variants):
        raise SystemExit("formation-bars must contain values of at least 90 bars")
    if spec["holdout_bars"] <= 0:
        raise SystemExit("holdout_bars must be positive")


def _execution_overrides(variant: dict) -> dict:
    allowed = {"signal_scale", "volatility_span_bars", "taker_fee_bps", "slippage_bps"}
    unknown = set(variant).difference(allowed | {"name"})
    if unknown:
        raise SystemExit(f"unknown signal variant fields: {sorted(unknown)}")
    return {key: value for key, value in variant.items() if key != "name"}


def _load_funding(
    connection: psycopg.Connection,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT symbol, funding_time, funding_rate, mark_price
            FROM market.funding_rate
            WHERE venue = 'binance' AND market_type = 'usdm_perpetual'
              AND symbol = ANY(%s) AND funding_time >= %s AND funding_time <= %s
            ORDER BY symbol, funding_time
            """,
            (symbols, start, end),
        )
        rows = cursor.fetchall()
    frame = pd.DataFrame(rows, columns=["symbol", "funding_time", "funding_rate", "mark_price"])
    if frame.empty:
        raise SystemExit("funding was requested but no funding events were found")
    result = {
        symbol: group.drop(columns="symbol").reset_index(drop=True)
        for symbol, group in frame.groupby("symbol")
    }
    missing = sorted(set(symbols).difference(result))
    if missing:
        raise SystemExit(f"funding was requested but symbols are missing events: {missing}")
    return result


if __name__ == "__main__":
    main()
