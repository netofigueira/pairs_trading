"""Cache narrow daily windows of the public historical Deribit option tape."""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from quant_pairs.history_deribit import HistoryDeribitClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="UTC date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--currency", default="BTC")
    parser.add_argument("--center-time", default="12:00:00")
    parser.add_argument("--window-minutes", type=int, default=120)
    parser.add_argument("--data-root", default="data/market/deribit/history-trades")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--request-delay-seconds", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.window_minutes <= 0 or args.request_delay_seconds < 0 or args.workers <= 0:
        parser.error("window, request delay and workers must be positive")
    start, end = _date(args.start), _date(args.end)
    if end < start:
        parser.error("end must not precede start")

    root = Path(args.data_root) / args.currency.upper() / "option"
    downloaded = skipped = total_trades = 0
    days = list(_days(start, end))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for day, status, trades, target in executor.map(
            lambda value: _collect_day(value, args, root), days
        ):
            if status == "skipped":
                skipped += 1
                continue
            downloaded += 1
            total_trades += trades
            print(f"date={day} trades={trades} path={target}")
    print(f"downloaded_days={downloaded} skipped_days={skipped} trades={total_trades}")


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from error


def _days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _collect_day(day: date, args: argparse.Namespace, root: Path) -> tuple[date, str, int, Path]:
    center = pd.Timestamp(f"{day.isoformat()}T{args.center_time}Z")
    target = root / day.isoformat() / f"{center.strftime('%H%M')}-{args.window_minutes}m.csv.gz"
    if target.exists() and target.stat().st_size > 0 and not args.overwrite:
        return day, "skipped", 0, target
    window = pd.Timedelta(minutes=args.window_minutes)
    trades = HistoryDeribitClient().option_trades(
        args.currency, start=center - window, end=center + window
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    trades.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(target)
    if args.request_delay_seconds:
        time.sleep(args.request_delay_seconds)
    return day, "downloaded", len(trades), target


if __name__ == "__main__":
    main()
