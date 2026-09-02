import pandas as pd

from quant_pairs.option_quotes import coverage_summary, round_trip_coverage


def test_round_trip_uses_ask_on_entry_and_bid_on_exit_for_same_contract() -> None:
    quotes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z", "2026-01-08T00:30:00Z"],
                utc=True,
            ),
            "instrument": ["BTC-30JAN26-100000-C", "BTC-30JAN26-90000-C", "BTC-30JAN26-100000-C"],
            "ask_price": [0.02, 0.03, 0.025],
            "bid_price": [0.019, 0.029, 0.024],
        }
    )

    matches = round_trip_coverage(
        quotes, horizon=pd.Timedelta(days=7), tolerance=pd.Timedelta(hours=1)
    )

    assert len(matches) == 1
    assert matches.loc[0, "instrument"] == "BTC-30JAN26-100000-C"
    assert matches.loc[0, "entry_ask"] == 0.02
    assert matches.loc[0, "exit_bid"] == 0.024
    assert coverage_summary(quotes, matches)["exact_contract_round_trips"] == 1
