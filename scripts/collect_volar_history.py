"""Collect a rate-limited grid of read-only Volar BTC chain snapshots."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from quant_pairs.data_lake import LocalDataLake
from quant_pairs.volar_api import VolarClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=13, help="UTC sandbox lookback, default: 13")
    parser.add_argument("--start", help="optional inclusive UTC ISO-8601 start")
    parser.add_argument("--end", help="optional inclusive UTC ISO-8601 end")
    parser.add_argument(
        "--interval-hours", type=int, default=12, help="snapshot cadence, default: 12"
    )
    parser.add_argument("--dotenv", default=".env", help="dotenv path; never printed")
    parser.add_argument("--data-root", default="data", help="local data lake root")
    parser.add_argument(
        "--min-request-seconds",
        type=float,
        default=6.1,
        help="minimum spacing; sandbox permits at most 10 requests/minute",
    )
    arguments = parser.parse_args()
    if arguments.days <= 0 or arguments.interval_hours <= 0:
        parser.error("--days and --interval-hours must be positive")
    if bool(arguments.start) != bool(arguments.end):
        parser.error("--start and --end must be supplied together")
    if arguments.min_request_seconds < 6:
        parser.error("--min-request-seconds must be at least 6 for the sandbox limit")
    return arguments


def main() -> None:
    arguments = parse_args()
    if arguments.start:
        start = pd.Timestamp(arguments.start)
        end = pd.Timestamp(arguments.end)
        if start.tzinfo is None or end.tzinfo is None:
            raise SystemExit("--start and --end must be timezone-aware UTC timestamps")
        start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
        if end < start:
            raise SystemExit("--end must be at or after --start")
    else:
        end = pd.Timestamp.now(tz="UTC").floor("h")
        start = end - pd.Timedelta(days=arguments.days)
    timestamps = pd.date_range(start, end, freq=f"{arguments.interval_hours}h", tz="UTC")
    client = VolarClient.from_environment(arguments.dotenv)
    lake = LocalDataLake(arguments.data_root)

    for number, timestamp in enumerate(timestamps, start=1):
        request_started = time.monotonic()
        chain = client.chain_snapshot("BTC", at=timestamp.isoformat())
        path = lake.write_option_chain_snapshot("volar", "BTC", chain)
        print(
            f"{number}/{len(timestamps)} timestamp={chain['timestamp'].iloc[0]} "
            f"rows={len(chain)} path={path}"
        )
        remaining = arguments.min_request_seconds - (time.monotonic() - request_started)
        if remaining > 0 and number < len(timestamps):
            time.sleep(remaining)


if __name__ == "__main__":
    main()
