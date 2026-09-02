"""Download declared first-of-month Tardis samples for the P1 plumbing gate."""

from __future__ import annotations

import argparse
from datetime import date

from quant_pairs.tardis import TardisDataError, download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2024-01")
    parser.add_argument("--end", default="2024-12")
    parser.add_argument("--data-root", default="data/market/tardis")
    parser.add_argument("--include-options-chain", action="store_true")
    arguments = parser.parse_args()
    months = _months(arguments.start, arguments.end)
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
                )
            except TardisDataError as error:
                print(f"date={sample_day} data_type={data_type} symbol={symbol} error={error}")
                continue
            print(f"date={sample_day} data_type={data_type} symbol={symbol} path={path}")


def _months(start: str, end: str) -> list[str]:
    start_year, start_month = _year_month(start)
    end_year, end_month = _year_month(end)
    if (end_year, end_month) < (start_year, start_month):
        raise ValueError("--end must not precede --start")
    months: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append(date(year, month, 1).strftime("%Y-%m"))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def _year_month(value: str) -> tuple[int, int]:
    try:
        year, month = (int(piece) for piece in value.split("-"))
        date(year, month, 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("month must use YYYY-MM") from error
    return year, month


if __name__ == "__main__":
    main()
