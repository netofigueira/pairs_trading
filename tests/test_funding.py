import pandas as pd
import pytest

from quant_pairs.funding import fetch_funding_history, funding_pnl_btc


def test_fetch_funding_history_pages_in_windows_and_normalises() -> None:
    calls = []

    def transport(endpoint, params):
        calls.append(params)
        start = int(params["start_timestamp"])
        return {
            "result": [
                {
                    "timestamp": start + 3_600_000,
                    "index_price": 42_000.0,
                    "interest_1h": 1e-5,
                    "interest_8h": 8e-5,
                }
            ]
        }

    frame = fetch_funding_history(
        "BTC-PERPETUAL",
        start=pd.Timestamp("2024-01-01T00:00:00Z"),
        end=pd.Timestamp("2024-03-01T00:00:00Z"),
        transport=transport,
    )

    assert len(calls) == 2  # 60 days paged in 30-day windows
    assert list(frame.columns) == ["timestamp", "index_price", "interest_1h", "interest_8h"]
    assert frame["timestamp"].dt.tz is not None
    assert len(frame) == 2


def test_funding_pnl_charges_longs_and_pays_shorts() -> None:
    start = pd.Timestamp("2024-01-01T12:00:00Z")
    funding = pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(hours=i) for i in range(1, 3)],
            "index_price": [40_000.0, 50_000.0],
            "interest_1h": [1e-4, 1e-4],
            "interest_8h": [8e-4, 8e-4],
        }
    )
    end = start + pd.Timedelta(hours=2)

    long_pnl = funding_pnl_btc(funding, contracts=100, start=start, end=end)
    short_pnl = funding_pnl_btc(funding, contracts=-100, start=start, end=end)

    expected = -(100 * 10 / 40_000 * 1e-4 + 100 * 10 / 50_000 * 1e-4)
    assert long_pnl == pytest.approx(expected)
    assert short_pnl == pytest.approx(-expected)


def test_funding_pnl_rejects_incomplete_coverage() -> None:
    start = pd.Timestamp("2024-01-01T12:00:00Z")
    funding = pd.DataFrame(
        {
            "timestamp": [start + pd.Timedelta(hours=1)],
            "index_price": [40_000.0],
            "interest_1h": [1e-4],
            "interest_8h": [8e-4],
        }
    )

    with pytest.raises(ValueError, match="1 of 8 hourly accruals"):
        funding_pnl_btc(
            funding, contracts=100, start=start, end=start + pd.Timedelta(hours=8)
        )
