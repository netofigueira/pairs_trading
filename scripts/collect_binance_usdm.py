"""Collect public Binance USDⓈ-M candles and funding into the local data lake."""

from __future__ import annotations

import argparse

import pandas as pd

from quant_pairs.binance_usdm import BinanceUSDMClient
from quant_pairs.data_lake import LocalDataLake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="USDⓈ-M symbol, e.g. BTCUSDT")
    parser.add_argument("--interval", default="1h", help="Binance kline interval, default: 1h")
    parser.add_argument("--days", type=int, default=30, help="UTC lookback in days, default: 30")
    parser.add_argument("--data-root", default="data", help="local data lake root, default: data")
    arguments = parser.parse_args()
    if arguments.days <= 0:
        parser.error("--days must be positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=arguments.days)
    client = BinanceUSDMClient()
    lake = LocalDataLake(arguments.data_root)

    klines = client.klines(arguments.symbol, arguments.interval, start=start, end=end)
    funding = client.funding_rates(arguments.symbol, start=start, end=end)
    kline_path = lake.upsert_klines("binance-usdm", arguments.symbol, arguments.interval, klines)
    funding_path = lake.upsert_funding("binance-usdm", arguments.symbol, funding)

    print(f"klines={len(klines)} path={kline_path}")
    print(f"funding_events={len(funding)} path={funding_path}")


if __name__ == "__main__":
    main()
