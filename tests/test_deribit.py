import pandas as pd
import pytest

from quant_pairs.deribit import (
    OPTION_SUMMARIES_ENDPOINT,
    VOLATILITY_INDEX_ENDPOINT,
    DeribitAPIError,
    DeribitClient,
)


def test_option_summaries_normalise_executable_fields() -> None:
    def transport(endpoint: str, params: dict[str, str | int]) -> dict:
        assert endpoint == OPTION_SUMMARIES_ENDPOINT
        assert params == {"currency": "BTC", "kind": "option"}
        return {
            "result": [
                {
                    "instrument_name": "BTC-30SEP26-100000-C",
                    "bid_price": 0.01,
                    "ask_price": 0.012,
                    "mark_price": 0.011,
                    "mark_iv": 48.5,
                    "underlying_price": 95_000.0,
                    "open_interest": 12.3,
                    "volume": 4.0,
                }
            ]
        }

    frame = DeribitClient(transport=transport).option_summaries(
        "btc", retrieved_at=pd.Timestamp("2026-09-01T12:00:00Z")
    )

    assert frame.loc[0, "instrument_name"] == "BTC-30SEP26-100000-C"
    assert frame.loc[0, "implied_volatility"] == 48.5
    assert frame.loc[0, "snapshot_time"] == pd.Timestamp("2026-09-01T12:00:00Z")


def test_volatility_index_normalises_utc_and_deduplicates() -> None:
    first = int(pd.Timestamp("2026-09-01T00:00:00Z").timestamp() * 1_000)

    def transport(endpoint: str, params: dict[str, str | int]) -> dict:
        assert endpoint == VOLATILITY_INDEX_ENDPOINT
        assert params["resolution"] == 3_600
        return {"result": {"data": [[first, 40, 42, 39, 41], [first, 40, 43, 39, 42]]}}

    frame = DeribitClient(transport=transport).volatility_index(
        "BTC",
        start=pd.Timestamp("2026-08-31T00:00:00Z"),
        end=pd.Timestamp("2026-09-02T00:00:00Z"),
    )

    assert len(frame) == 1
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2026-09-01T00:00:00Z")
    assert frame.loc[0, "close"] == 42


def test_option_summaries_reject_missing_executable_quote_fields() -> None:
    def transport(_: str, __: dict[str, str | int]) -> dict:
        return {"result": [{"instrument_name": "BTC-30SEP26-100000-C"}]}

    with pytest.raises(DeribitAPIError, match="missing fields"):
        DeribitClient(transport=transport).option_summaries("BTC")


def test_volatility_history_follows_the_continuation_cursor() -> None:
    calls: list[int] = []

    def transport(_: str, params: dict[str, str | int]) -> dict:
        cursor = int(params["end_timestamp"])
        calls.append(cursor)
        if len(calls) == 1:
            return {"result": {"data": [[2_000, 0.4, 0.5, 0.3, 0.45]], "continuation": 1_500}}
        return {"result": {"data": [[1_000, 0.3, 0.4, 0.2, 0.35]], "continuation": None}}

    frame = DeribitClient(transport=transport).volatility_index_history(
        "BTC",
        start=pd.Timestamp("1970-01-01T00:00:01Z"),
        end=pd.Timestamp("1970-01-01T00:00:02Z"),
    )

    assert calls == [2_000, 1_500]
    assert frame["close"].tolist() == [0.35, 0.45]


def test_chart_data_normalises_ohlcv_arrays() -> None:
    def transport(_: str, __: dict[str, str | int]) -> dict:
        return {
            "result": {
                "ticks": [1_000],
                "open": [100],
                "high": [110],
                "low": [90],
                "close": [105],
                "volume": [12],
            }
        }

    frame = DeribitClient(transport=transport).chart_data(
        "BTC-PERPETUAL",
        start=pd.Timestamp("1970-01-01T00:00:00Z"),
        end=pd.Timestamp("1970-01-01T00:00:02Z"),
    )

    assert frame.loc[0, "timestamp"] == pd.Timestamp("1970-01-01T00:00:01Z")
    assert frame.loc[0, "close"] == 105
