"""Download declared first-of-month Tardis samples for the P1 plumbing gate."""

from __future__ import annotations

import argparse

from quant_pairs.tardis import TardisDataError, download_dataset, sample_months


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", default="2024-12")
    parser.add_argument("--data-root", default="data/market/tardis")
    parser.add_argument("--include-options-chain", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--month-step",
        type=int,
        default=1,
        help="sample every Nth month from --start (use 3 for a quarterly pilot)",
    )
    arguments = parser.parse_args()
    months = sample_months(arguments.start, arguments.end, step=arguments.month_step)
    datasets = [("quotes", "OPTIONS"), ("quotes", "BTC-PERPETUAL")]
    if arguments.include_options_chain:
        datasets.append(("options_chain", "OPTIONS"))
    for month in months:
        sample_day = f"{month}-01"
        for data_type, symbol in datasets:
            try:
                path = download_dataset(
                    arguments.data_root,
                    exchange="deribit",
                    data_type=data_type,
                    date=sample_day,
                    symbol=symbol,
                    overwrite=arguments.overwrite,
                )
            except TardisDataError as error:
                print(f"date={sample_day} data_type={data_type} symbol={symbol} error={error}")
                continue
            print(f"date={sample_day} data_type={data_type} symbol={symbol} path={path}")
if __name__ == "__main__":
    main()
