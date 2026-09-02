"""Reconstruct and summarize final executable books from a Tardis quote CSV.GZ."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from quant_pairs.tardis_quotes import reconstruct_top_of_book


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--max-age-seconds", type=int, default=300)
    arguments = parser.parse_args()
    if arguments.max_age_seconds < 0:
        parser.error("--max-age-seconds cannot be negative")
    books = reconstruct_top_of_book(
        arguments.path,
        chunk_rows=arguments.chunk_rows,
        max_age=pd.Timedelta(seconds=arguments.max_age_seconds),
    )
    print(
        json.dumps(
            {
                "executable_books": len(books),
                "max_age_seconds": arguments.max_age_seconds,
                "first_timestamp": str(books["timestamp"].min()) if not books.empty else None,
                "last_timestamp": str(books["timestamp"].max()) if not books.empty else None,
                "median_relative_spread_bps": (
                    float(((books["ask_price"] / books["bid_price"] - 1) * 10_000).median())
                    if not books.empty
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
