"""Read-only client for the public historical Deribit trade tape.

``history.deribit.com`` is distinct from the main public API: the latter keeps
trade queries for only 24 hours, while this host exposes an indexed historical
tape.  The service signals a truncated response with ``has_more`` but does not
return a continuation cursor, so this client recursively bisects a time window
until every request fits in one response.  It never advances a timestamp cursor
and therefore cannot silently drop multiple trades sharing one millisecond.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

HISTORY_DERIBIT_BASE_URL = "https://history.deribit.com/api/v2"
HISTORY_TRADES_ENDPOINT = "/public/get_last_trades_by_currency_and_time"
_MAX_PAGE_SIZE = 1_000

HistoryTransport = Callable[[str, Mapping[str, str | int]], dict[str, Any]]


class HistoryDeribitAPIError(RuntimeError):
    """The historical trade host returned an unusable response."""


class HistoryDeribitClient:
    """Fetch complete UTC intervals of public historical option trades."""

    def __init__(self, transport: HistoryTransport | None = None) -> None:
        self._transport = transport or _history_http_transport

    def option_trades(
        self,
        currency: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return all BTC/ETH option trades in a closed millisecond interval.

        ``end`` is inclusive at the API boundary.  The recursively partitioned
        requests are deduplicated by the venue's currency-wide ``trade_id``.
        """

        start_ms, end_ms = _utc_milliseconds(start), _utc_milliseconds(end)
        if end_ms < start_ms:
            raise ValueError("end must not precede start")
        rows = self._fetch_complete_range(currency.upper(), start_ms, end_ms)
        if not rows:
            return _empty_trade_frame()
        frame = pd.DataFrame(rows)
        required = {"trade_id", "timestamp", "price", "instrument_name", "direction"}
        missing = required.difference(frame.columns)
        if missing:
            raise HistoryDeribitAPIError(
                f"historical trade response is missing fields: {sorted(missing)}"
            )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        for column in ("price", "mark_price", "iv", "index_price", "amount", "contracts"):
            if column not in frame:
                frame[column] = float("nan")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["currency"] = currency.upper()
        frame["source"] = "history_deribit_public_trades"
        columns = [
            "timestamp",
            "currency",
            "trade_id",
            "trade_seq",
            "instrument_name",
            "price",
            "mark_price",
            "iv",
            "index_price",
            "amount",
            "contracts",
            "direction",
            "tick_direction",
            "liquidation",
            "source",
        ]
        for column in columns:
            if column not in frame:
                frame[column] = pd.NA
        return (
            frame.loc[:, columns]
            .drop_duplicates(subset="trade_id", keep="last")
            .sort_values(["timestamp", "trade_id"])
            .reset_index(drop=True)
        )

    def _fetch_complete_range(self, currency: str, start_ms: int, end_ms: int) -> list[dict]:
        payload = self._transport(
            HISTORY_TRADES_ENDPOINT,
            {
                "currency": currency,
                "kind": "option",
                "start_timestamp": start_ms,
                "end_timestamp": end_ms,
                "count": _MAX_PAGE_SIZE,
                "sorting": "asc",
            },
        )
        result = _result(payload)
        if not isinstance(result, dict) or not isinstance(result.get("trades"), list):
            raise HistoryDeribitAPIError("historical trade response is missing result.trades")
        trades = result["trades"]
        if not result.get("has_more", False):
            return trades

        # A continuation cursor is not supplied by this public historical host.
        # Bisecting preserves all same-millisecond trades at the price of more
        # requests on unusually busy days.
        if start_ms >= end_ms:
            raise HistoryDeribitAPIError(
                "more than 1000 trades share one millisecond; cannot collect losslessly"
            )
        middle = start_ms + (end_ms - start_ms) // 2
        if middle == end_ms:
            middle -= 1
        return self._fetch_complete_range(currency, start_ms, middle) + self._fetch_complete_range(
            currency, middle + 1, end_ms
        )


def _history_http_transport(endpoint: str, params: Mapping[str, str | int]) -> dict[str, Any]:
    url = f"{HISTORY_DERIBIT_BASE_URL}{endpoint}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=60) as response:  # noqa: S310 - fixed public host
            payload = json.load(response)
    except OSError as error:
        message = f"history.deribit request failed for {endpoint}: {error}"
        raise HistoryDeribitAPIError(message) from error
    if not isinstance(payload, dict):
        raise HistoryDeribitAPIError("historical trade response is not a JSON object")
    return payload


def _result(payload: dict[str, Any]) -> Any:
    if "error" in payload:
        raise HistoryDeribitAPIError(f"history.deribit returned {payload['error']}")
    if "result" not in payload:
        raise HistoryDeribitAPIError("historical trade response is missing result")
    return payload["result"]


def _utc_milliseconds(value: pd.Timestamp) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return int(timestamp.tz_convert("UTC").timestamp() * 1_000)


def _empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "currency",
            "trade_id",
            "trade_seq",
            "instrument_name",
            "price",
            "mark_price",
            "iv",
            "index_price",
            "amount",
            "contracts",
            "direction",
            "tick_direction",
            "liquidation",
            "source",
        ]
    )
