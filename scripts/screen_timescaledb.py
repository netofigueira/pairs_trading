"""Run and persist one FDR-controlled formation screen from TimescaleDB."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import psycopg
from psycopg.types.json import Json

from quant_pairs.screener import screen_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="config/universe.crypto-usdm-v1.json")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--formation-bars", type=int, default=4_320)
    args = parser.parse_args()
    database_url = os.environ["QUANT_PAIRS_DATABASE_URL"]
    universe = json.loads(Path(args.universe).read_text())
    symbols = universe["symbols"]
    with psycopg.connect(database_url) as connection:
        prices = _load_prices(connection, symbols, args.interval, args.formation_bars)
        results = screen_pairs(prices)
        run_id = _persist(connection, prices, args, universe, results)
    accepted = [result for result in results if result.accepted]
    print(f"formation_run={run_id} pairs={len(results)} accepted={len(accepted)}")
    for result in results[:20]:
        print(
            f"{result.model.dependent}/{result.model.independent} "
            f"p={result.model.coint_pvalue:.5f} q={result.fdr_qvalue:.5f} "
            f"half_life={result.half_life_bars} accepted={result.accepted}"
        )


def _load_prices(
    connection: psycopg.Connection,
    symbols: list[str],
    interval: str,
    bars: int,
    *,
    require_bars: bool = True,
) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT open_time, symbol, close
            FROM market.candle
            WHERE venue = 'binance' AND market_type = 'usdm_perpetual'
              AND interval = %s AND symbol = ANY(%s) AND close_time <= now()
            ORDER BY open_time DESC
            LIMIT %s
            """,
            (interval, symbols, bars * len(symbols)),
        )
        rows = cursor.fetchall()
    frame = pd.DataFrame(rows, columns=["open_time", "symbol", "close"])
    prices = frame.pivot(index="open_time", columns="symbol", values="close").sort_index().dropna()
    if require_bars and len(prices) < bars:
        raise SystemExit(f"need {bars} common closed bars; received {len(prices)}")
    return prices.iloc[-bars:]


def _persist(
    connection: psycopg.Connection,
    prices: pd.DataFrame,
    args: argparse.Namespace,
    universe: dict,
    results: list,
) -> str:
    config = {
        "universe": universe["name"],
        "interval": args.interval,
        "formation_bars": args.formation_bars,
        "fdr_alpha": 0.05,
        "half_life_bars": [4, 72],
    }
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO research.formation_run (config, data_start, data_end, completed_at)
            VALUES (%s, %s, %s, now()) RETURNING id
            """,
            (Json(config), prices.index.min(), prices.index.max()),
        )
        run_id = cursor.fetchone()[0]
        for result in results:
            model = result.model
            diagnostics = Json(
                {"fdr_qvalue": result.fdr_qvalue, "critical_values": model.critical_values}
            )
            cursor.execute(
                """
                INSERT INTO research.candidate (
                    formation_run_id, dependent_symbol, independent_symbol, hedge_alpha,
                    hedge_beta, coint_t_stat, coint_pvalue, half_life_bars, accepted, diagnostics
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    run_id, model.dependent, model.independent, model.alpha, model.beta,
                    model.coint_t_stat, model.coint_pvalue, result.half_life_bars,
                    result.accepted, diagnostics,
                ),
            )
    return str(run_id)


if __name__ == "__main__":
    main()
