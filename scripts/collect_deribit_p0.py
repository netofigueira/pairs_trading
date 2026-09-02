"""Collect the public, daily inputs for the Deribit IV-versus-RV P0 study."""

from __future__ import annotations

import argparse

import pandas as pd

from quant_pairs.data_lake import LocalDataLake
from quant_pairs.deribit import DeribitClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01T00:00:00Z")
    parser.add_argument("--end", help="UTC end; defaults to the current completed UTC day")
    parser.add_argument("--data-root", default="data")
    arguments = parser.parse_args()
    start = _utc(arguments.start)
    end = _utc(arguments.end) if arguments.end else pd.Timestamp.now(tz="UTC").floor("D")
    if end <= start:
        raise SystemExit("--end must be after --start")

    client = DeribitClient()
    lake = LocalDataLake(arguments.data_root)
    dvol = client.volatility_index_history("BTC", start=start, end=end, resolution="1D")
    perp = client.chart_data("BTC-PERPETUAL", start=start, end=end, resolution="1D")
    dvol_path = lake.upsert_volatility_index("deribit", "BTC", dvol)
    perp_path = lake.upsert_price_bars("deribit", "BTC-PERPETUAL", "1D", perp)
    print(f"dvol_bars={len(dvol)} path={dvol_path}")
    print(f"perpetual_bars={len(perp)} path={perp_path}")


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise SystemExit("timestamps must be timezone-aware UTC values")
    return timestamp.tz_convert("UTC")


if __name__ == "__main__":
    main()
