"""Run the executable intraday straddle plumbing gate on one Tardis sample day."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_pairs.tardis_intraday import run_intraday_straddle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2024-01-01")
    parser.add_argument("--entry-time", default="12:00:00")
    parser.add_argument("--exit-time", default="20:00:00")
    parser.add_argument("--data-root", default="data/market/tardis")
    parser.add_argument("--max-age-seconds", type=int, default=300)
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=30)
    parser.add_argument("--contracts", type=float, default=1.0)
    parser.add_argument("--with-options-chain", action="store_true")
    parser.add_argument("--perp-taker-fee-rate", type=float, default=0.0005)
    arguments = parser.parse_args()
    root = Path(arguments.data_root) / "deribit" / "quotes" / arguments.date
    result = run_intraday_straddle(
        root / "OPTIONS.csv.gz",
        root / "BTC-PERPETUAL.csv.gz",
        entry_at=pd.Timestamp(f"{arguments.date}T{arguments.entry_time}Z"),
        exit_at=pd.Timestamp(f"{arguments.date}T{arguments.exit_time}Z"),
        max_age=pd.Timedelta(seconds=arguments.max_age_seconds),
        min_dte=arguments.min_dte,
        max_dte=arguments.max_dte,
        contracts=arguments.contracts,
        options_chain_path=(
            Path(arguments.data_root)
            / "deribit"
            / "options_chain"
            / arguments.date
            / "OPTIONS.csv.gz"
            if arguments.with_options_chain
            else None
        ),
        perp_taker_fee_rate=arguments.perp_taker_fee_rate,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
