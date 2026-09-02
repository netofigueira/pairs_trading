import pandas as pd

from quant_pairs.data_lake import LocalDataLake


def test_upsert_klines_merges_and_deduplicates(tmp_path) -> None:
    lake = LocalDataLake(tmp_path)
    first = pd.DataFrame(
        {
            "open_time": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "close_time": [pd.Timestamp("2025-01-01T00:00:59Z")],
            "close": [100.0],
        }
    )
    second = pd.DataFrame(
        {
            "open_time": [
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-01T00:01:00Z"),
            ],
            "close_time": [
                pd.Timestamp("2025-01-01T00:00:59Z"),
                pd.Timestamp("2025-01-01T00:01:59Z"),
            ],
            "close": [101.0, 102.0],
        }
    )

    path = lake.upsert_klines("binance-usdm", "btcusdt", "1m", first)
    lake.upsert_klines("binance-usdm", "btcusdt", "1m", second)
    stored = pd.read_csv(path)

    assert path.name == "1m.csv.gz"
    assert stored["close"].tolist() == [101.0, 102.0]


def test_upsert_funding_accepts_mixed_iso8601_precision(tmp_path) -> None:
    lake = LocalDataLake(tmp_path)
    first = pd.DataFrame(
        {
            "funding_time": ["2025-01-01T00:00:00+00:00"],
            "funding_rate": [0.0001],
            "mark_price": [100.0],
        }
    )
    second = pd.DataFrame(
        {
            "funding_time": ["2025-01-01T08:00:00.001+00:00"],
            "funding_rate": [0.0002],
            "mark_price": [101.0],
        }
    )

    lake.upsert_funding("binance-usdm", "btcusdt", first)
    path = lake.upsert_funding("binance-usdm", "btcusdt", second)

    assert len(pd.read_csv(path)) == 2


def test_upsert_option_summaries_replaces_same_snapshot_and_instrument(tmp_path) -> None:
    lake = LocalDataLake(tmp_path)
    snapshot = pd.Timestamp("2026-09-01T12:00:00Z")
    first = pd.DataFrame(
        {
            "snapshot_time": [snapshot],
            "instrument_name": ["BTC-30SEP26-100000-C"],
            "bid_price": [0.01],
            "ask_price": [0.012],
            "implied_volatility": [48.0],
        }
    )
    second = first.assign(ask_price=0.011, implied_volatility=47.0)

    lake.upsert_option_summaries("deribit", "BTC", first)
    path = lake.upsert_option_summaries("deribit", "BTC", second)
    stored = pd.read_csv(path)

    assert len(stored) == 1
    assert stored.loc[0, "ask_price"] == 0.011


def test_upsert_volatility_index_replaces_same_timestamp(tmp_path) -> None:
    lake = LocalDataLake(tmp_path)
    timestamp = pd.Timestamp("2026-09-01T12:00:00Z")
    first = pd.DataFrame({"timestamp": [timestamp], "close": [42.0]})

    lake.upsert_volatility_index("deribit", "BTC", first)
    path = lake.upsert_volatility_index("deribit", "BTC", first.assign(close=43.0))

    assert pd.read_csv(path).loc[0, "close"] == 43.0


def test_write_option_chain_snapshot_requires_one_timestamp(tmp_path) -> None:
    lake = LocalDataLake(tmp_path)
    timestamp = pd.Timestamp("2026-08-25T12:00:00Z")
    frame = pd.DataFrame(
        {
            "timestamp": [timestamp],
            "instrument": ["BTC-29AUG26-100000-C"],
            "bid_price": [0.01],
            "ask_price": [0.012],
            "source": ["live_ws"],
        }
    )

    path = lake.write_option_chain_snapshot("volar", "BTC", frame)

    assert path.name == "20260825T120000Z.parquet"
    assert pd.read_parquet(path).loc[0, "instrument"] == "BTC-29AUG26-100000-C"
