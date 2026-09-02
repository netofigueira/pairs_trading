"""Public Deribit funding history and inverse-perp funding accounting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pairs.deribit import DeribitAPIError, Transport, _http_transport, _result

FUNDING_ENDPOINT = "/public/get_funding_rate_history"
PERP_CONTRACT_SIZE_USD = 10.0
_MAX_WINDOW = pd.Timedelta(days=30)


def fetch_funding_history(
    instrument_name: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    transport: Transport | None = None,
) -> pd.DataFrame:
    """Return hourly funding observations over a UTC interval, paged in 30-day windows."""

    start_utc, end_utc = _utc(start), _utc(end)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    transport = transport or _http_transport
    pages: list[pd.DataFrame] = []
    cursor = start_utc
    while cursor < end_utc:
        window_end = min(cursor + _MAX_WINDOW, end_utc)
        payload = transport(
            FUNDING_ENDPOINT,
            {
                "instrument_name": instrument_name,
                "start_timestamp": int(cursor.timestamp() * 1_000),
                "end_timestamp": int(window_end.timestamp() * 1_000),
            },
        )
        result = _result(payload, FUNDING_ENDPOINT)
        if not isinstance(result, list):
            raise DeribitAPIError("funding history result is not a list")
        pages.append(_normalise_rows(result))
        cursor = window_end
    frame = (
        pd.concat(pages, ignore_index=True)
        .drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return frame


def load_funding_history(
    instrument_name: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_path: Path | str,
    transport: Transport | None = None,
) -> pd.DataFrame:
    """Fetch funding history through a local CSV cache keyed by the full interval."""

    path = Path(cache_path)
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["timestamp"])
        cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True)
        if not cached.empty and (
            cached["timestamp"].iloc[0] <= _utc(start) + pd.Timedelta(hours=1)
            and cached["timestamp"].iloc[-1] >= _utc(end) - pd.Timedelta(hours=1)
        ):
            return cached
    frame = fetch_funding_history(
        instrument_name, start=start, end=end, transport=transport
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def funding_pnl_btc(
    funding: pd.DataFrame,
    *,
    contracts: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    contract_size_usd: float = PERP_CONTRACT_SIZE_USD,
) -> float:
    """Funding P&L in BTC for an inverse-perp position held over (start, end].

    Longs pay positive hourly interest; shorts receive it. Position value in BTC
    is re-marked at each hourly index print, matching Deribit's continuous accrual.
    """

    if funding.empty:
        raise ValueError("funding history is empty")
    start_utc, end_utc = _utc(start), _utc(end)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    window = funding.loc[
        (funding["timestamp"] > start_utc) & (funding["timestamp"] <= end_utc)
    ]
    expected_hours = int((end_utc - start_utc) / pd.Timedelta(hours=1))
    if len(window) < expected_hours:
        raise ValueError(
            f"funding history covers {len(window)} of {expected_hours} hourly accruals"
        )
    position_btc = contracts * contract_size_usd / window["index_price"]
    return float(-(position_btc * window["interest_1h"]).sum())


def _normalise_rows(rows: list[object]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "index_price", "interest_1h", "interest_8h"])
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    for column in ("index_price", "interest_1h", "interest_8h"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")
