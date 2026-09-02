"""Public, read-only Deribit options market-data client.

This module deliberately exposes no account, authentication, or order methods.
It is the data boundary for the volatility-research track.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"
OPTION_SUMMARIES_ENDPOINT = "/public/get_book_summary_by_currency"
VOLATILITY_INDEX_ENDPOINT = "/public/get_volatility_index_data"
CHART_ENDPOINT = "/public/get_tradingview_chart_data"

Transport = Callable[[str, Mapping[str, str | int]], dict[str, Any]]


class DeribitAPIError(RuntimeError):
    """The public Deribit endpoint returned an invalid or unusable response."""


class DeribitClient:
    """Read public option summaries and DVOL history with UTC normalization."""

    def __init__(self, transport: Transport | None = None) -> None:
        self._transport = transport or _http_transport

    def option_summaries(
        self, currency: str, *, retrieved_at: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Return one executable-price snapshot for every option in a currency."""

        payload = self._transport(
            OPTION_SUMMARIES_ENDPOINT, {"currency": currency.upper(), "kind": "option"}
        )
        result = _result(payload, OPTION_SUMMARIES_ENDPOINT)
        if not isinstance(result, list) or not all(isinstance(row, dict) for row in result):
            raise DeribitAPIError("option summary result is not a list of objects")
        snapshot_time = pd.Timestamp.now(tz="UTC") if retrieved_at is None else _utc(retrieved_at)
        frame = pd.DataFrame(result)
        if frame.empty:
            return _empty_option_summaries()
        required = {"instrument_name", "bid_price", "ask_price", "mark_price", "mark_iv"}
        missing = required.difference(frame.columns)
        if missing:
            raise DeribitAPIError(f"option summary is missing fields: {sorted(missing)}")
        output = pd.DataFrame(
            {
                "snapshot_time": snapshot_time,
                "currency": currency.upper(),
                "instrument_name": frame["instrument_name"].astype("string"),
                "bid_price": pd.to_numeric(frame["bid_price"], errors="raise"),
                "ask_price": pd.to_numeric(frame["ask_price"], errors="raise"),
                "mark_price": pd.to_numeric(frame["mark_price"], errors="raise"),
                "implied_volatility": pd.to_numeric(frame["mark_iv"], errors="raise"),
                "underlying_price": _numeric_or_missing(frame, "underlying_price"),
                "open_interest": _numeric_or_missing(frame, "open_interest"),
                "volume": _numeric_or_missing(frame, "volume"),
            }
        )
        return output.sort_values("instrument_name").reset_index(drop=True)

    def volatility_index(
        self,
        currency: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        resolution: int = 3_600,
    ) -> pd.DataFrame:
        """Return historical DVOL OHLC observations over a UTC interval."""

        if resolution <= 0:
            raise ValueError("resolution must be positive")
        start_ms, end_ms = _utc_milliseconds(start), _utc_milliseconds(end)
        if end_ms <= start_ms:
            raise ValueError("end must be after start")
        payload = self._transport(
            VOLATILITY_INDEX_ENDPOINT,
            {
                "currency": currency.upper(),
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "resolution": resolution,
            },
        )
        result = _result(payload, VOLATILITY_INDEX_ENDPOINT)
        if not isinstance(result, dict) or not isinstance(result.get("data"), list):
            raise DeribitAPIError("volatility index result is missing data")
        frame = pd.DataFrame(result["data"], columns=["timestamp", "open", "high", "low", "close"])
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "currency", "open", "high", "low", "close"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame["currency"] = currency.upper()
        for column in ("open", "high", "low", "close"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return (
            frame.drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def volatility_index_history(
        self,
        currency: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        resolution: str = "1D",
    ) -> pd.DataFrame:
        """Page backwards through public DVOL candles without silently truncating history."""

        start_ms, cursor = _utc_milliseconds(start), _utc_milliseconds(end)
        if cursor <= start_ms:
            raise ValueError("end must be after start")
        pages: list[pd.DataFrame] = []
        while cursor > start_ms:
            payload = self._transport(
                VOLATILITY_INDEX_ENDPOINT,
                {
                    "currency": currency.upper(),
                    "start_timestamp": start_ms,
                    "end_timestamp": cursor,
                    "resolution": resolution,
                },
            )
            result = _result(payload, VOLATILITY_INDEX_ENDPOINT)
            if not isinstance(result, dict) or not isinstance(result.get("data"), list):
                raise DeribitAPIError("volatility index result is missing data")
            pages.append(_normalise_dvol_rows(result["data"], currency))
            continuation = result.get("continuation")
            if continuation is None:
                break
            next_cursor = int(continuation)
            if next_cursor >= cursor:
                raise DeribitAPIError("volatility index continuation did not move backwards")
            cursor = next_cursor
        return (
            pd.concat(pages, ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def chart_data(
        self,
        instrument_name: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        resolution: str = "1D",
    ) -> pd.DataFrame:
        """Return public OHLCV chart bars for an instrument."""

        payload = self._transport(
            CHART_ENDPOINT,
            {
                "instrument_name": instrument_name,
                "start_timestamp": _utc_milliseconds(start),
                "end_timestamp": _utc_milliseconds(end),
                "resolution": resolution,
            },
        )
        result = _result(payload, CHART_ENDPOINT)
        if not isinstance(result, dict) or not isinstance(result.get("ticks"), list):
            raise DeribitAPIError("chart result is missing ticks")
        columns = ("timestamp", "open", "high", "low", "close", "volume")
        arrays = {"timestamp": result["ticks"]}
        arrays.update({column: result.get(column, []) for column in columns[1:]})
        lengths = {column: len(values) for column, values in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise DeribitAPIError(f"chart arrays have inconsistent lengths: {lengths}")
        frame = pd.DataFrame(arrays)
        if frame.empty:
            return pd.DataFrame(columns=columns)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        for column in columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        return frame.sort_values("timestamp").reset_index(drop=True)


def _http_transport(endpoint: str, params: Mapping[str, str | int]) -> dict[str, Any]:
    url = f"{DERIBIT_BASE_URL}{endpoint}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=30) as response:  # noqa: S310 - fixed official base URL
            payload = json.load(response)
    except OSError as error:
        raise DeribitAPIError(f"Deribit request failed for {endpoint}: {error}") from error
    if not isinstance(payload, dict):
        raise DeribitAPIError("Deribit response is not a JSON object")
    return payload


def _result(payload: dict[str, Any], endpoint: str) -> Any:
    if "error" in payload:
        raise DeribitAPIError(f"Deribit returned {payload['error']} for {endpoint}")
    if "result" not in payload:
        raise DeribitAPIError(f"Deribit response is missing result for {endpoint}")
    return payload["result"]


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC values")
    return timestamp.tz_convert("UTC")


def _utc_milliseconds(value: pd.Timestamp) -> int:
    return int(_utc(value).timestamp() * 1_000)


def _empty_option_summaries() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "snapshot_time",
            "currency",
            "instrument_name",
            "bid_price",
            "ask_price",
            "mark_price",
            "implied_volatility",
            "underlying_price",
            "open_interest",
            "volume",
        ]
    )


def _normalise_dvol_rows(rows: list[Any], currency: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "currency", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["currency"] = currency.upper()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _numeric_or_missing(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")
