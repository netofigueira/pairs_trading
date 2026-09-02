import pandas as pd

from quant_pairs.rolling_volatility_book import (
    RollingBookParameters,
    run_synthetic_rolling_short_book,
)


def _daily(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2026-01-01 08:00", periods=len(values), freq="1D", tz="UTC"
            ),
            "close": values,
        }
    )


def test_rolling_book_only_enters_short_signals_and_respects_capacity() -> None:
    prices = _daily([100.0] * 35)
    dvol = _daily([60.0] * 35)
    forecasts = pd.DataFrame(
        {
            "forecast_at": pd.date_range("2026-01-04 08:00", periods=18, freq="1D", tz="UTC"),
            # Most days are short signals against a 59% synthetic bid IV; the
            # final non-signal also checks that entries are signal-gated.
            "garch_corrected_rv": [0.50] * 17 + [0.70],
        }
    )
    result = run_synthetic_rolling_short_book(
        prices,
        dvol,
        forecasts,
        parameters=RollingBookParameters(
            horizon_days=4,
            contracts_per_entry=0.25,
            max_contracts_per_btc=0.5,
            bid_iv_discount_points=1.0,
        ),
    )

    assert result["coverage"]["short_signals"] > result["coverage"]["accepted_entries"]
    assert result["coverage"]["skipped_for_book_capacity"] > 0
    assert result["summary"]["max_gross_option_contracts"] <= 0.5 * max(
        row["equity_btc"] for row in result["daily"]
    )
    assert all(entry["forecast_rv"] < entry["synthetic_bid_iv"] for entry in result["entries"])


def test_rolling_book_settles_all_positions_and_reports_daily_margin() -> None:
    prices = _daily([100.0, 105.0, 95.0, 103.0, 100.0, 101.0, 99.0, 100.0] * 5)
    dvol = _daily([60.0] * len(prices))
    forecasts = pd.DataFrame(
        {
            "forecast_at": pd.date_range("2026-01-04 08:00", periods=25, freq="1D", tz="UTC"),
            "garch_corrected_rv": [0.30] * 25,
        }
    )
    result = run_synthetic_rolling_short_book(
        prices,
        dvol,
        forecasts,
        parameters=RollingBookParameters(horizon_days=3, contracts_per_entry=0.1),
    )

    assert result["entries"]
    assert result["daily"][-1]["active_positions"] == 0
    assert all("maintenance_margin_btc" in row for row in result["daily"])
