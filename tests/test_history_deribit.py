import pandas as pd

from quant_pairs.history_deribit import HISTORY_TRADES_ENDPOINT, HistoryDeribitClient


def _trade(trade_id: str, timestamp: int) -> dict:
    return {
        "trade_id": trade_id,
        "trade_seq": 1,
        "timestamp": timestamp,
        "price": 0.05,
        "mark_price": 0.051,
        "iv": 60.0,
        "instrument_name": "BTC-26JUN20-10000-C",
        "index_price": 9000.0,
        "direction": "buy",
        "amount": 1.0,
    }


def test_history_client_normalizes_public_trade_fields() -> None:
    def transport(endpoint: str, params: dict[str, str | int]) -> dict:
        assert endpoint == HISTORY_TRADES_ENDPOINT
        assert params["kind"] == "option"
        assert params["count"] == 1000
        return {"result": {"trades": [_trade("1", 1_000)], "has_more": False}}

    frame = HistoryDeribitClient(transport=transport).option_trades(
        "btc",
        start=pd.Timestamp("1970-01-01T00:00:01Z"),
        end=pd.Timestamp("1970-01-01T00:00:01Z"),
    )

    assert frame.loc[0, "timestamp"] == pd.Timestamp("1970-01-01T00:00:01Z")
    assert frame.loc[0, "iv"] == 60.0
    assert frame.loc[0, "source"] == "history_deribit_public_trades"


def test_history_client_bisects_truncated_ranges_without_dropping_boundary_trades() -> None:
    calls: list[tuple[int, int]] = []

    def transport(_: str, params: dict[str, str | int]) -> dict:
        start, end = int(params["start_timestamp"]), int(params["end_timestamp"])
        calls.append((start, end))
        if end - start > 1:
            return {"result": {"trades": [_trade("ignored", start)], "has_more": True}}
        return {
            "result": {
                "trades": [_trade(str(start), start), _trade(str(end), end)],
                "has_more": False,
            }
        }

    frame = HistoryDeribitClient(transport=transport).option_trades(
        "BTC",
        start=pd.Timestamp("1970-01-01T00:00:00Z"),
        end=pd.Timestamp("1970-01-01T00:00:00.003Z"),
    )

    assert calls[0] == (0, 3)
    assert frame["trade_id"].tolist() == ["0", "1", "2", "3"]
