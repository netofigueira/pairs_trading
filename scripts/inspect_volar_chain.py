"""Validate a historical Volar option-chain Parquet file for executable backtests."""

from __future__ import annotations

import argparse
import json

from quant_pairs.volar import load_executable_chain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="path to a Volar chain Parquet file")
    arguments = parser.parse_args()
    result = load_executable_chain(arguments.path)
    quotes = result.quotes
    print(
        json.dumps(
            {
                "source_rows": result.source_rows,
                "executable_rows": result.executable_rows,
                "rejected_rows": result.rejected_rows,
                "snapshots": int(quotes["timestamp"].nunique()),
                "instruments": int(quotes["instrument"].nunique()),
                "start": str(quotes["timestamp"].min()) if not quotes.empty else None,
                "end": str(quotes["timestamp"].max()) if not quotes.empty else None,
                "sources": sorted(quotes["source"].unique().tolist()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
