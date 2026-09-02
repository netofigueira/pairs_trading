"""Audit whether local historical option quotes support an executable P1 study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_pairs.option_quotes import coverage_summary, load_quote_snapshots, round_trip_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains-dir", default="data/market/volar/options/BTC/chains")
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--tolerance-hours", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.horizon_days <= 0 or arguments.tolerance_hours < 0:
        parser.error("horizon must be positive and tolerance cannot be negative")
    paths = list(Path(arguments.chains_dir).glob("*.parquet"))
    quotes = load_quote_snapshots(paths)
    matches = round_trip_coverage(
        quotes,
        horizon=pd.Timedelta(days=arguments.horizon_days),
        tolerance=pd.Timedelta(hours=arguments.tolerance_hours),
    )
    summary = coverage_summary(quotes, matches)
    summary["horizon_days"] = arguments.horizon_days
    summary["gate"] = (
        "insufficient for a historical strategy backtest"
        if summary["coverage_days"] < 180
        else "eligible for strategy specification review"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
