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
