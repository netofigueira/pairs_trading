"""Load cached Deribit historical option-trade files into TimescaleDB."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg

_COLUMNS = (
    "timestamp", "currency", "trade_id", "trade_seq", "instrument_name", "price",
    "mark_price", "iv", "index_price", "amount", "contracts", "direction",
    "tick_direction", "liquidation", "source",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/market/deribit/history-trades/BTC/option")
    parser.add_argument("--start", help="UTC date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", help="UTC date, inclusive (YYYY-MM-DD)")
    args = parser.parse_args()
    database_url = os.environ.get("QUANT_PAIRS_DATABASE_URL")
    if not database_url:
        raise SystemExit("set QUANT_PAIRS_DATABASE_URL before loading the tape")

    files = _files(Path(args.data_root), args.start, args.end)
    loaded = skipped = inserted = 0
    with psycopg.connect(database_url) as connection:
        for path in files:
            result = _load_file(connection, path)
            if result is None:
                skipped += 1
            else:
                loaded += 1
                inserted += result
                print(f"file={path} inserted_rows={result}")
    print(f"loaded_files={loaded} skipped_files={skipped} inserted_rows={inserted}")


def _files(root: Path, start: str | None, end: str | None) -> list[Path]:
    files = sorted(root.glob("*/1200-120m.csv.gz"))
    if start:
        files = [path for path in files if path.parent.name >= start]
    if end:
        files = [path for path in files if path.parent.name <= end]
    return files


def _load_file(connection: psycopg.Connection[Any], path: Path) -> int | None:
    digest = _sha256(path)
    source_path = str(path)
    with connection.transaction(), connection.cursor() as cursor:
        cursor.execute(
            "SELECT sha256 FROM market.tape_ingestion_file WHERE source_path = %s", (source_path,)
        )
        existing = cursor.fetchone()
        if existing and existing[0] == digest:
            return None

        frame = pd.read_csv(path, compression="gzip", parse_dates=["timestamp"])
        missing = set(_COLUMNS).difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["timestamp"] = _timestamps(frame["timestamp"])
        rows = [_row(row) for row in frame.loc[:, _COLUMNS].itertuples(index=False, name=None)]
        cursor.execute(
            "CREATE TEMP TABLE option_trade_stage "
            "(LIKE market.option_trade INCLUDING DEFAULTS) ON COMMIT DROP"
        )
        with cursor.copy(
            "COPY option_trade_stage (traded_at, currency, trade_id, trade_seq, instrument_name, "
            "price, mark_price, iv, index_price, amount, contracts, direction, tick_direction, "
            "liquidation, source) FROM STDIN"
        ) as copy:
            for row in rows:
                copy.write_row(row)
        cursor.execute(
            """
            INSERT INTO market.option_trade (
                traded_at, currency, trade_id, trade_seq, instrument_name, price, mark_price, iv,
                index_price, amount, contracts, direction, tick_direction, liquidation, source
            )
            SELECT traded_at, currency, trade_id, trade_seq, instrument_name, price, mark_price, iv,
                   index_price, amount, contracts, direction, tick_direction, liquidation, source
            FROM option_trade_stage
            ON CONFLICT DO NOTHING
            """
        )
        inserted = cursor.rowcount
        first_at = frame["timestamp"].min().to_pydatetime() if not frame.empty else None
        last_at = frame["timestamp"].max().to_pydatetime() if not frame.empty else None
        cursor.execute(
            """
            INSERT INTO market.tape_ingestion_file (
                source_path, sha256, source_first_at, source_last_at, source_rows, loaded_rows
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (source_path) DO UPDATE SET
                sha256 = EXCLUDED.sha256,
                source_first_at = EXCLUDED.source_first_at,
                source_last_at = EXCLUDED.source_last_at,
                source_rows = EXCLUDED.source_rows,
                loaded_rows = EXCLUDED.loaded_rows,
                loaded_at = now()
            """,
            (source_path, digest, first_at, last_at, len(frame), inserted),
        )
    return inserted


def _row(row: tuple[object, ...]) -> tuple[object, ...]:
    return tuple(_value(value) for value in row)


def _value(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def _timestamps(values: pd.Series) -> pd.Series:
    """Normalize CSV timestamps whose optional fractional seconds vary by row."""
    return pd.to_datetime(values, utc=True, format="ISO8601")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
