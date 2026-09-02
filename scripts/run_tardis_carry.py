"""Run the hold-to-expiry straddle carry gate on free Tardis monthly sample days.

For each entry date: buy the ATM straddle at real Tardis asks, hold to expiry,
settle at the official Deribit delivery price, and (with --with-options-chain)
carry a static delta hedge charged with public hourly funding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_pairs.funding import load_funding_history
from quant_pairs.settlement import load_delivery_prices
from quant_pairs.tardis_carry import run_carry_straddle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        action="append",
        dest="dates",
        help="entry date (repeatable); defaults to every date under the quotes root",
    )
    parser.add_argument("--entry-time", default="12:00:00")
    parser.add_argument("--data-root", default="data/market/tardis")
    parser.add_argument("--cache-root", default="data/market/deribit")
    parser.add_argument("--max-age-seconds", type=int, default=300)
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=30)
    parser.add_argument("--contracts", type=float, default=1.0)
    parser.add_argument("--with-options-chain", action="store_true")
    parser.add_argument("--perp-taker-fee-rate", type=float, default=0.0005)
    parser.add_argument("--target-dte", type=float, default=14.0)
    parser.add_argument("--hedge-exit-slippage-bps", type=float, default=0.0)
    arguments = parser.parse_args()

    quotes_root = Path(arguments.data_root) / "deribit" / "quotes"
    dates = arguments.dates or sorted(
        path.name for path in quotes_root.iterdir() if path.is_dir()
    )
    if not dates:
        raise SystemExit(f"no Tardis sample days under {quotes_root}")
    cache_root = Path(arguments.cache_root)

    results: list[dict[str, object]] = []
    for date in dates:
        entry_at = pd.Timestamp(f"{date}T{arguments.entry_time}Z")
        day_root = quotes_root / date
        delivery_prices = load_delivery_prices(
            "btc_usd",
            cache_path=cache_root / "delivery_prices" / "btc_usd.csv",
            required_date=entry_at + pd.Timedelta(days=arguments.max_dte + 1),
        )
        funding = None
        if arguments.with_options_chain:
            funding = load_funding_history(
                "BTC-PERPETUAL",
                start=entry_at,
                end=entry_at + pd.Timedelta(days=arguments.max_dte + 1),
                cache_path=cache_root / "funding" / f"BTC-PERPETUAL-{date}.csv",
            )
        try:
            result = run_carry_straddle(
                day_root / "OPTIONS.csv.gz",
                day_root / "BTC-PERPETUAL.csv.gz",
                entry_at=entry_at,
                delivery_prices=delivery_prices,
                max_age=pd.Timedelta(seconds=arguments.max_age_seconds),
                min_dte=arguments.min_dte,
                max_dte=arguments.max_dte,
                contracts=arguments.contracts,
                options_chain_path=(
                    Path(arguments.data_root)
                    / "deribit"
                    / "options_chain"
                    / date
                    / "OPTIONS.csv.gz"
                    if arguments.with_options_chain
                    else None
                ),
                funding=funding,
                perp_taker_fee_rate=arguments.perp_taker_fee_rate,
                target_dte=arguments.target_dte,
                hedge_exit_slippage_bps=arguments.hedge_exit_slippage_bps,
            )
        except (ValueError, FileNotFoundError) as error:
            result = {"status": "failed", "entry_date": date, "error": str(error)}
        results.append(result)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
