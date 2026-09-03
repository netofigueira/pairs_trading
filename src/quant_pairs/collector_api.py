"""Private API invoked by n8n to perform idempotent market-data collection."""

from __future__ import annotations

import os
import secrets
from datetime import UTC

import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .binance_usdm import BinanceUSDMClient
from .history_deribit import HistoryDeribitClient
from .timescale import TimescaleDataStore

app = FastAPI(title="Quant collector", docs_url=None, redoc_url=None)


class CollectRequest(BaseModel):
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    interval: str = "1h"
    initial_lookback_days: int = Field(default=365, ge=1, le=3650)
    backfill_days: int | None = Field(
        default=None,
        ge=1,
        le=3650,
        description="explicit historical range; bypasses the incremental cursor",
    )


class CollectTapeRequest(BaseModel):
    currencies: list[str] = Field(default_factory=lambda: ["BTC", "ETH"])
    initial_lookback_hours: int = Field(default=24, ge=1, le=8_760)
    backfill_hours: int | None = Field(
        default=None,
        ge=1,
        le=8_760,
        description="explicit historical range; bypasses the incremental cursor",
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/collect")
def collect(
    request: CollectRequest, x_collector_token: str = Header(default="")
) -> dict[str, object]:
    _authorize(x_collector_token)
    database_url = _required_env("QUANT_PAIRS_DATABASE_URL")
    store = TimescaleDataStore(database_url)
    client = BinanceUSDMClient()
    now = pd.Timestamp.now(tz=UTC).floor("min")
    result: dict[str, object] = {"interval": request.interval, "symbols": {}}

    for symbol in request.symbols:
        start = (
            now - pd.Timedelta(days=request.backfill_days)
            if request.backfill_days is not None
            else _incremental_start(
                store, symbol, request.interval, now, request.initial_lookback_days
            )
        )
        klines = client.klines(symbol, request.interval, start=start, end=now)
        funding = client.funding_rates(symbol, start=start, end=now)
        result["symbols"][symbol] = {
            "from": start.isoformat(),
            "klines": store.upsert_klines(symbol, request.interval, klines),
            "funding": store.upsert_funding(symbol, funding),
        }
    return result


@app.post("/v1/collect-tape")
def collect_tape(
    request: CollectTapeRequest, x_collector_token: str = Header(default="")
) -> dict[str, object]:
    _authorize(x_collector_token)
    database_url = _required_env("QUANT_PAIRS_DATABASE_URL")
    store = TimescaleDataStore(database_url)
    client = HistoryDeribitClient()
    now = pd.Timestamp.now(tz=UTC)
    result: dict[str, object] = {"currencies": {}}

    for currency in request.currencies:
        start = (
            now - pd.Timedelta(hours=request.backfill_hours)
            if request.backfill_hours is not None
            else _incremental_tape_start(store, currency, now, request.initial_lookback_hours)
        )
        trades = client.option_trades(currency, start=start, end=now)
        result["currencies"][currency] = {
            "from": start.isoformat(),
            "trades": store.upsert_option_trades(currency, trades),
        }
    return result


def _incremental_tape_start(
    store: TimescaleDataStore, currency: str, now: pd.Timestamp, initial_lookback_hours: int
) -> pd.Timestamp:
    latest = store.latest_option_trade_time(currency)
    if latest is None:
        return now - pd.Timedelta(hours=initial_lookback_hours)
    return latest - pd.Timedelta(minutes=5)


def _incremental_start(
    store: TimescaleDataStore,
    symbol: str,
    interval: str,
    now: pd.Timestamp,
    initial_lookback_days: int,
) -> pd.Timestamp:
    latest = store.latest_kline_time(symbol, interval)
    if latest is None:
        return now - pd.Timedelta(days=initial_lookback_days)
    return latest - (2 * _interval_delta(interval))


def _interval_delta(interval: str) -> pd.Timedelta:
    try:
        return pd.Timedelta(interval)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"unsupported interval: {interval}") from error


def _authorize(provided: str) -> None:
    expected = _required_env("QUANT_COLLECTOR_TOKEN")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="invalid collector token")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value
