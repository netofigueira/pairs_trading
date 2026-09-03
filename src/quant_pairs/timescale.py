"""Persistence adapter for the private TimescaleDB market data store."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pandas as pd


class TimescaleDataStore:
    """Writes public market data without granting any trading capability."""

    venue = "binance"
    market_type = "usdm_perpetual"

    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
        except ImportError as error:  # pragma: no cover
            message = "install project dependencies with `pip install -e '.[dev]'`"
            raise RuntimeError(message) from error
        self._psycopg = psycopg
        self._database_url = database_url

    def upsert_klines(self, symbol: str, interval: str, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        records = [
            (
                self.venue, self.market_type, symbol, interval, _timestamp(row.open_time),
                _timestamp(row.close_time), _numeric(row.open), _numeric(row.high),
                _numeric(row.low), _numeric(row.close), _numeric(row.volume),
                _numeric(row.quote_volume), int(row.trade_count),
                _numeric(row.taker_buy_base_volume), _numeric(row.taker_buy_quote_volume),
            )
            for row in data.itertuples(index=False)
        ]
        with self._psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                self._upsert_instrument(cursor, symbol)
                cursor.executemany(_CANDLE_UPSERT, records)
        return len(records)

    def latest_kline_time(self, symbol: str, interval: str) -> pd.Timestamp | None:
        with self._psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT max(open_time)
                    FROM market.candle
                    WHERE venue = %s AND market_type = %s AND symbol = %s AND interval = %s
                    """,
                    (self.venue, self.market_type, symbol, interval),
                )
                value = cursor.fetchone()[0]
        return pd.Timestamp(value) if value is not None else None

    def upsert_funding(self, symbol: str, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        records = [
            (self.venue, self.market_type, symbol, _timestamp(row.funding_time),
             _numeric(row.funding_rate), _numeric(row.mark_price))
            for row in data.itertuples(index=False)
        ]
        with self._psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                self._upsert_instrument(cursor, symbol)
                cursor.executemany(_FUNDING_UPSERT, records)
        return len(records)

    def upsert_option_trades(self, currency: str, data: pd.DataFrame) -> int:
        if data.empty:
            return 0
        records = [
            (
                currency, _timestamp(row.timestamp), str(row.trade_id),
                _int_or_none(row.trade_seq), row.instrument_name, _numeric(row.price),
                _numeric(row.mark_price), _numeric(row.iv), _numeric(row.index_price),
                _numeric(row.amount), _numeric(row.contracts), row.direction,
                _int_or_none(row.tick_direction), _text_or_none(row.liquidation), row.source,
            )
            for row in data.itertuples(index=False)
        ]
        with self._psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(_OPTION_TRADE_UPSERT, records)
        return len(records)

    def latest_option_trade_time(self, currency: str) -> pd.Timestamp | None:
        with self._psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT max(traded_at) FROM market.option_trade WHERE currency = %s",
                    (currency,),
                )
                value = cursor.fetchone()[0]
        return pd.Timestamp(value) if value is not None else None

    def _upsert_instrument(self, cursor: object, symbol: str) -> None:
        cursor.execute(
            """
            INSERT INTO market.instrument (venue, market_type, symbol, base_asset, quote_asset)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (venue, market_type, symbol) DO UPDATE
            SET updated_at = now(), active = TRUE
            """,
            (self.venue, self.market_type, symbol, _base_asset(symbol), "USDT"),
        )


def _timestamp(value: object) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def _numeric(value: object) -> Decimal | None:
    if pd.isna(value):
        return None
    return Decimal(str(value))


def _base_asset(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def _int_or_none(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _text_or_none(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


_CANDLE_UPSERT = """
INSERT INTO market.candle (
    venue, market_type, symbol, interval, open_time, close_time, open, high, low, close,
    volume, quote_volume, trade_count, taker_buy_base_volume, taker_buy_quote_volume
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (venue, market_type, symbol, interval, open_time) DO UPDATE SET
    close_time = EXCLUDED.close_time, open = EXCLUDED.open, high = EXCLUDED.high,
    low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume,
    quote_volume = EXCLUDED.quote_volume, trade_count = EXCLUDED.trade_count,
    taker_buy_base_volume = EXCLUDED.taker_buy_base_volume,
    taker_buy_quote_volume = EXCLUDED.taker_buy_quote_volume,
    source_ingested_at = now()
"""

_FUNDING_UPSERT = """
INSERT INTO market.funding_rate (
    venue, market_type, symbol, funding_time, funding_rate, mark_price
) VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (venue, market_type, symbol, funding_time) DO UPDATE SET
    funding_rate = EXCLUDED.funding_rate, mark_price = EXCLUDED.mark_price,
    source_ingested_at = now()
"""

_OPTION_TRADE_UPSERT = """
INSERT INTO market.option_trade (
    currency, traded_at, trade_id, trade_seq, instrument_name, price, mark_price, iv,
    index_price, amount, contracts, direction, tick_direction, liquidation, source
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (venue, currency, trade_id, traded_at) DO NOTHING
"""
