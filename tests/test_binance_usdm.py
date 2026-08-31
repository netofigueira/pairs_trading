from datetime import UTC

import pandas as pd

from quant_pairs.binance_usdm import FUNDING_ENDPOINT, KLINES_ENDPOINT, BinanceUSDMClient


def utc(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def test_klines_paginate_and_normalise_to_utc() -> None:
    first_open = int(utc("2025-01-01T00:00:00").timestamp() * 1_000)
    first_close = first_open + 59_999
    second_open = first_close + 1
    second_close = second_open + 59_999
    calls: list[int] = []

    def transport(endpoint: str, params: dict[str, str | int]) -> list[object]:
        assert endpoint == KLINES_ENDPOINT
        calls.append(int(params["startTime"]))
        if len(calls) == 1:
            return [[first_open, "1", "2", "0.5", "1.5", "10", first_close, "15", 2, "4", "6", "0"]]
        if len(calls) == 2:
            return [[second_open, "1.5", "3", "1", "2", "12", second_close, "20", 3, "5", "8", "0"]]
        return []

    frame = BinanceUSDMClient(transport=transport).klines(
        "btcusdt",
        "1m",
        start=utc("2025-01-01T00:00:00"),
        end=utc("2025-01-01T00:03:00"),
    )

    assert calls == [first_open, second_open, second_close + 1]
    assert frame["open_time"].dt.tz == UTC
    assert frame["close"].tolist() == [1.5, 2.0]


def test_funding_paginates_and_converts_numeric_fields() -> None:
    first = int(utc("2025-01-01T00:00:00").timestamp() * 1_000)
    second = int(utc("2025-01-01T08:00:00").timestamp() * 1_000)
    calls = 0

    def transport(endpoint: str, params: dict[str, str | int]) -> list[object]:
        nonlocal calls
        assert endpoint == FUNDING_ENDPOINT
        calls += 1
        if calls == 1:
            return [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": first,
                    "fundingRate": "0.0001",
                    "markPrice": "100",
                }
            ]
        if calls == 2:
            return [
                {
                    "symbol": "BTCUSDT",
                    "fundingTime": second,
                    "fundingRate": "-0.0002",
                    "markPrice": "101",
                }
            ]
        return []

    frame = BinanceUSDMClient(transport=transport).funding_rates(
        "BTCUSDT",
        start=utc("2025-01-01T00:00:00"),
        end=utc("2025-01-02T00:00:00"),
    )

    assert frame["funding_rate"].tolist() == [0.0001, -0.0002]
    assert frame["mark_price"].tolist() == [100.0, 101.0]


def test_klines_exclude_a_candle_that_has_not_closed() -> None:
    open_time = int(utc("2025-01-01T00:00:00").timestamp() * 1_000)
    close_time = open_time + 59_999

    def transport(_: str, __: dict[str, str | int]) -> list[object]:
        return [[open_time, "1", "2", "0.5", "1.5", "10", close_time, "15", 2, "4", "6", "0"]]

    frame = BinanceUSDMClient(transport=transport).klines(
        "BTCUSDT",
        "1m",
        start=utc("2025-01-01T00:00:00"),
        end=utc("2025-01-01T00:00:30"),
    )

    assert frame.empty
