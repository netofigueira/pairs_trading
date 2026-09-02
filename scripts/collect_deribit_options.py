"""Collect public Deribit option quotes and DVOL history into the local research lake."""

from __future__ import annotations

import argparse

import pandas as pd

from quant_pairs.data_lake import LocalDataLake
from quant_pairs.deribit import DeribitClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--currency",
        action="append",
        choices=("BTC", "ETH"),
        required=True,
        help="currency to collect; repeat for BTC and ETH",
    )
    parser.add_argument("--dvol-days", type=int, default=30, help="DVOL lookback in UTC days")
    parser.add_argument("--data-root", default="data", help="local data lake root")
    arguments = parser.parse_args()
    if arguments.dvol_days <= 0:
        parser.error("--dvol-days must be positive")
    return arguments


def main() -> None:
    arguments = parse_args()
    client = DeribitClient()
    lake = LocalDataLake(arguments.data_root)
    end = pd.Timestamp.now(tz="UTC").floor("min")
    start = end - pd.Timedelta(days=arguments.dvol_days)

    for currency in dict.fromkeys(arguments.currency):
        summaries = client.option_summaries(currency, retrieved_at=end)
        dvol = client.volatility_index(currency, start=start, end=end)
        summary_path = lake.upsert_option_summaries("deribit", currency, summaries)
        dvol_path = lake.upsert_volatility_index("deribit", currency, dvol)
        print(f"currency={currency} option_summaries={len(summaries)} path={summary_path}")
        print(f"currency={currency} dvol_bars={len(dvol)} path={dvol_path}")


if __name__ == "__main__":
    main()
