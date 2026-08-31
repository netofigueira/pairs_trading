"""Public, read-only Binance USDⓈ-M market-data client.

This adapter deliberately has no authentication or order-placement capability.
It is safe to use for research data collection and is kept separate from a
future execution adapter.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

BINANCE_USDM_BASE_URL = "https://fapi.binance.com"
KLINES_ENDPOINT = "/fapi/v1/klines"
FUNDING_ENDPOINT = "/fapi/v1/fundingRate"
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)

Transport = Callable[[str, Mapping[str, str | int]], list[Any]]


class BinanceAPIError(RuntimeError):
    """The public Binance endpoint returned an invalid or unusable response."""


class BinanceUSDMClient:
    """Paginated market-data client for USDⓈ-M futures.

    All input datetimes are normalized to UTC milliseconds. Returned timestamps
    are timezone-aware UTC values, which prevents accidental local-time joins.
    """

    def __init__(self, transport: Transport | None = None) -> None:
        self._transport = transport or _http_transport

    def klines(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch every closed kline from start through end, with pagination."""

        start_ms = _utc_milliseconds(start)
        end_ms = _utc_milliseconds(end)
        if end_ms <= start_ms:
            raise ValueError("end must be after start")

        rows: list[list[Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            page = self._transport(
                KLINES_ENDPOINT,
                {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1_500,
                },
            )
            if not page:
                break
            if not isinstance(page[0], list):
                raise BinanceAPIError("kline response is not a list of rows")
            rows.extend(page)
            next_cursor = int(page[-1][6]) + 1
            if next_cursor <= cursor:
                raise BinanceAPIError("kline pagination did not advance")
            cursor = next_cursor

        frame = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        if frame.empty:
            return _empty_klines()
        frame = frame.drop_duplicates(subset="open_time", keep="last")
        frame = frame[frame["open_time"].astype("int64") < end_ms].copy()
        return _normalise_klines(frame)

    def funding_rates(
        self,
        symbol: str,
        *,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
    ) -> pd.DataFrame:
        """Fetch funding history with the same UTC and pagination guarantees."""

        start_ms = _utc_milliseconds(start)
        end_ms = _utc_milliseconds(end)
        if end_ms <= start_ms:
            raise ValueError("end must be after start")

        rows: list[dict[str, Any]] = []
        cursor = start_ms
        while cursor < end_ms:
            page = self._transport(
                FUNDING_ENDPOINT,
                {
                    "symbol": symbol.upper(),
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": 1_000,
                },
            )
            if not page:
                break
            if not isinstance(page[0], dict):
                raise BinanceAPIError("funding response is not a list of objects")
            rows.extend(page)
            next_cursor = int(page[-1]["fundingTime"]) + 1
            if next_cursor <= cursor:
                raise BinanceAPIError("funding pagination did not advance")
            cursor = next_cursor

        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(
                columns=["symbol", "funding_time", "funding_rate", "mark_price"]
            ).astype({"symbol": "string", "funding_rate": "float64", "mark_price": "float64"})
        frame = frame.rename(
            columns={
                "fundingTime": "funding_time",
                "fundingRate": "funding_rate",
                "markPrice": "mark_price",
            }
        )
        frame = frame.drop_duplicates(subset="funding_time", keep="last")
        frame["funding_time"] = pd.to_datetime(frame["funding_time"], unit="ms", utc=True)
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="raise")
        frame["mark_price"] = pd.to_numeric(frame["mark_price"], errors="raise")
        return frame.sort_values("funding_time").reset_index(drop=True)


def _http_transport(endpoint: str, params: Mapping[str, str | int]) -> list[Any]:
    url = f"{BINANCE_USDM_BASE_URL}{endpoint}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed official base URL
            payload = json.load(response)
    except OSError as error:
        raise BinanceAPIError(f"Binance request failed for {endpoint}: {error}") from error
    if isinstance(payload, dict) and "code" in payload:
        raise BinanceAPIError(f"Binance returned {payload}")
    if not isinstance(payload, list):
        raise BinanceAPIError("Binance response is not a JSON list")
    return payload


def _utc_milliseconds(value: datetime | pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC values")
    return int(timestamp.tz_convert("UTC").timestamp() * 1_000)


def _normalise_klines(frame: pd.DataFrame) -> pd.DataFrame:
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit="ms", utc=True)
    numeric_columns = [
        column for column in KLINE_COLUMNS if column not in {"open_time", "close_time", "ignore"}
    ]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
    frame = frame.drop(columns="ignore").sort_values("open_time").reset_index(drop=True)
    return frame


def _empty_klines() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[column for column in KLINE_COLUMNS if column != "ignore"]
    )
